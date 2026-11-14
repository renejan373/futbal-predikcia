"""Pomocné funkcie na skladanie dokumentu .docx.

Skripty v tomto priečinku nie sú súčasťou modelovacieho pipeline – slúžia len
na regenerovanie textu práce z výsledkov v `reports/tables` a `reports/figures`.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx.document import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph

BODY_FONT = "Times New Roman"
BODY_SIZE = Pt(12)
TABLE_SIZE = Pt(9)
CAPTION_SIZE = Pt(10)


def set_run_font(run, size: Pt | None = None, bold: bool | None = None) -> None:
    run.font.name = BODY_FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    if size is not None:
        run.font.size = size
    if bold is not None:
        run.bold = bold


def add_rich_runs(paragraph: Paragraph, text: str, size: Pt | None = None) -> None:
    """Do odstavca vloží text s podporou **tučného** zvýraznenia."""
    # Spätné apostrofy z Markdownu sa v tlačenej práci nezobrazujú.
    text = text.replace("`", "")
    for index, chunk in enumerate(re.split(r"(\*\*[^*]+\*\*)", text)):
        if not chunk:
            continue
        bold = chunk.startswith("**") and chunk.endswith("**")
        run = paragraph.add_run(chunk[2:-2] if bold else chunk)
        set_run_font(run, size=size, bold=bold if bold else None)


class DocumentBuilder:
    """Vkladá odstavce, tabuľky a obrázky pred zadanú kotvu."""

    def __init__(self, document: Document, anchor: Paragraph) -> None:
        self.document = document
        self.anchor = anchor

    # ------------------------------------------------------------------
    # Základné bloky
    # ------------------------------------------------------------------
    def _place(self, element) -> None:
        self.anchor._p.addprevious(element)

    def heading(self, text: str, level: int = 1) -> Paragraph:
        paragraph = self.document.add_paragraph(style=f"Heading {level}")
        paragraph.add_run(text)
        self._place(paragraph._p)
        return paragraph

    def paragraph(
        self,
        text: str = "",
        indent: bool = True,
        size: Pt | None = None,
        align: WD_ALIGN_PARAGRAPH | None = None,
        style: str = "Normal",
        space_after: Pt | None = None,
        keep_with_next: bool = False,
    ) -> Paragraph:
        paragraph = self.document.add_paragraph(style=style)
        add_rich_runs(paragraph, text, size=size)
        if indent:
            paragraph.paragraph_format.first_line_indent = Cm(0.76)
        if align is not None:
            paragraph.paragraph_format.alignment = align
        if space_after is not None:
            paragraph.paragraph_format.space_after = space_after
        if keep_with_next:
            paragraph.paragraph_format.keep_with_next = True
        self._place(paragraph._p)
        return paragraph

    def bullet(self, text: str, numbered: bool = False) -> Paragraph:
        paragraph = self.document.add_paragraph(style="List Paragraph")
        add_rich_runs(paragraph, text)
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.left_indent = Cm(0.9)
        paragraph.paragraph_format.space_after = Pt(3)
        fmt = paragraph.paragraph_format
        fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        self._place(paragraph._p)
        return paragraph

    def page_break(self) -> None:
        paragraph = self.document.add_paragraph()
        paragraph.add_run().add_break(WD_BREAK.PAGE)
        self._place(paragraph._p)

    def caption(
        self,
        text: str,
        above: bool = True,
    ) -> Paragraph:
        paragraph = self.document.add_paragraph()
        run = paragraph.add_run(text)
        set_run_font(run, size=CAPTION_SIZE)
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(6 if above else 3)
        paragraph.paragraph_format.space_after = Pt(6 if above else 12)
        if above:
            paragraph.paragraph_format.keep_with_next = True
        self._place(paragraph._p)
        return paragraph

    # ------------------------------------------------------------------
    # Tabuľky a obrázky
    # ------------------------------------------------------------------
    def table(
        self,
        caption: str,
        headers: list[str],
        rows: list[list[object]],
        widths: list[float] | None = None,
        align: str = "center",
    ) -> Table:
        self.caption(caption, above=True)

        table = self.document.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True

        header_cells = table.rows[0].cells
        for cell, header in zip(header_cells, headers):
            cell.text = ""
            paragraph = cell.paragraphs[0]
            run = paragraph.add_run(header)
            set_run_font(run, size=TABLE_SIZE, bold=True)
            paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.first_line_indent = Cm(0)
            paragraph.paragraph_format.space_after = Pt(1)
            shd = OxmlElement("w:shd")
            shd.set(qn("w:fill"), "E7E9EC")
            cell._tc.get_or_add_tcPr().append(shd)

        for row in rows:
            cells = table.add_row().cells
            for index, (cell, value) in enumerate(zip(cells, row)):
                cell.text = ""
                paragraph = cell.paragraphs[0]
                text = value if isinstance(value, str) else str(value)
                run = paragraph.add_run(text)
                set_run_font(run, size=TABLE_SIZE)
                paragraph.paragraph_format.first_line_indent = Cm(0)
                paragraph.paragraph_format.space_after = Pt(1)
                if index > 0:
                    paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.RIGHT if align == "center" and _is_number(text) else WD_ALIGN_PARAGRAPH.CENTER

        if widths:
            for row in table.rows:
                for cell, width in zip(row.cells, widths):
                    cell.width = Cm(width)

        # Hlavička sa opakuje na novej strane a riadok sa nikdy nerozdelí.
        for row in table.rows:
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_before = Pt(1)
        table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        # Tabuľku držíme pokope s popisom (okrem posledného riadka).
        for row in table.rows[:-1]:
            for cell in row.cells:
                cell.paragraphs[0].paragraph_format.keep_with_next = True

        self._place(table._tbl)
        # Medzera za tabuľkou.
        spacer = self.document.add_paragraph()
        spacer.paragraph_format.space_after = Pt(6)
        spacer.paragraph_format.line_spacing = 1.0
        self._place(spacer._p)
        return table

    def figure(self, path: str | Path, caption: str, width_cm: float = 14.5) -> None:
        paragraph = self.document.add_paragraph()
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.keep_with_next = True
        run = paragraph.add_run()
        run.add_picture(str(path), width=Cm(width_cm))
        self._place(paragraph._p)
        self.caption(caption, above=False)

    def toc_field(self) -> None:
        """Vloží pole Obsahu; Word ho pri otvorení dokumentu aktualizuje."""
        paragraph = self.document.add_paragraph()
        paragraph.paragraph_format.first_line_indent = Cm(0)
        run = paragraph.add_run()
        begin = OxmlElement("w:fldChar")
        begin.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = ' TOC \\o "1-3" \\h \\z \\u '
        separate = OxmlElement("w:fldChar")
        separate.set(qn("w:fldCharType"), "separate")
        text = OxmlElement("w:t")
        text.text = "Obsah sa doplní po aktualizovaní polí (Ctrl+A, F9)."
        end = OxmlElement("w:fldChar")
        end.set(qn("w:fldCharType"), "end")
        for element in (begin, instr, separate, text, end):
            run._r.append(element)
        self._place(paragraph._p)


def _is_number(text: str) -> bool:
    return bool(re.fullmatch(r"[+\-−]?[\d\s.,%:]+", text.strip()))


def enable_field_updates(document: Document) -> None:
    """Nastaví, aby Word pri otvorení aktualizoval polia (obsah)."""
    settings = document.settings.element
    update = settings.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        # `w:updateFields` musí v settings.xml stáť pred neskoršími elementmi,
        # inak Word dokument označí za poškodený.
        later = (
            "w:compat",
            "w:rsids",
            "m:mathPr",
            "w:themeFontLang",
            "w:clrSchemeMapping",
            "w:shapeDefaults",
            "w:decimalSymbol",
            "w:listSeparator",
        )
        successor = None
        for tag in later:
            successor = settings.find(qn(tag))
            if successor is not None:
                break
        if successor is not None:
            successor.addprevious(update)
        else:
            settings.append(update)
    update.set(qn("w:val"), "true")


def clear_between(
    document: Document,
    start_text: str,
    end_text: str,
    skip_texts: tuple[str, ...] = (),
) -> Paragraph:
    """Odstráni odstavce medzi dvoma kotvami a vráti koncovú kotvu.

    Odstavce, ktoré obsahujú zalomenie sekcie, sa zachovajú – inak by sa
    rozbilo číslovanie strán a okraje.
    """
    paragraphs = document.paragraphs
    start = next(i for i, p in enumerate(paragraphs) if p.text.strip() == start_text)
    end = next(
        i
        for i, p in enumerate(paragraphs)
        if i > start and p.text.strip() == end_text
    )

    for paragraph in paragraphs[start:end]:
        if "sectPr" in paragraph._p.xml:
            continue
        parent = paragraph._p.getparent()
        parent.remove(paragraph._p)

    return document.paragraphs[
        next(i for i, p in enumerate(document.paragraphs) if p.text.strip() == end_text)
    ]
