"""Fill the official ETSINF ODS template from anexo-ods.tex: the 17 marks and the
reflection. Run again whenever the .tex changes.

    uv run python thesis/anexo-ods-docx.py
"""
import re
import xml.dom.minidom
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs/research/writing/ODS en TFG-TFM de titulaciones ETSINF-Anexo.docx"
DST = ROOT / "thesis/anexo-ods.docx"
TEX = ROOT / "thesis/capitulos/anexo-ods.tex"

RPR = ('<w:rPr><w:rFonts w:asciiTheme="minorHAnsi" w:hAnsiTheme="minorHAnsi" '
       'w:cstheme="minorHAnsi"/>{i}<w:szCs w:val="21"/><w:lang w:val="es-ES"/></w:rPr>')
MARK = ('<w:r><w:rPr><w:rFonts w:asciiTheme="minorHAnsi" w:hAnsiTheme="minorHAnsi" '
        'w:cstheme="minorHAnsi"/><w:b/><w:lang w:val="es-ES"/></w:rPr><w:t>X</w:t></w:r>')
SPACING = '<w:spacing w:before="60" w:after="60"/>'


def levels(tex: str) -> dict[str, int]:
    """ODS name -> column index (0 Alto, 1 Medio, 2 Bajo, 3 No procede), from the table."""
    table = tex.split(r"\midrule")[1].split(r"\bottomrule")[0]
    out = {}
    for line in table.strip().splitlines():
        name, *cells = [c.strip() for c in line.rstrip("\\").split("&")]
        marks = [i for i, c in enumerate(cells) if c]
        assert len(marks) == 1, line
        out[re.sub(r"^\d+\.\s*", "", name)] = marks[0]
    assert len(out) == 17
    return out


def runs(paragraph: str) -> list[tuple[str, bool]]:
    """(text, italic) runs of one LaTeX paragraph; only \\emph and a few glyphs are allowed."""
    p = paragraph.replace("\\,\\%", " %").replace("~", " ")
    out, pos = [], 0
    for m in re.finditer(r"\\emph\{([^}]*)\}", p):
        out += [(p[pos:m.start()], False), (m.group(1), True)]
        pos = m.end()
    out.append((p[pos:], False))
    assert "\\" not in "".join(t for t, _ in out), paragraph
    return [(t, i) for t, i in out if t]


def para_xml(paragraph: str) -> str:
    r = "".join(
        f'<w:r>{RPR.format(i="<w:i/>" if it else "")}<w:t xml:space="preserve">{escape(t)}</w:t></w:r>'
        for t, it in runs(paragraph)
    )
    return f'<w:p><w:pPr><w:spacing w:after="120"/><w:jc w:val="both"/>{RPR.format(i="")}</w:pPr>{r}</w:p>'


def fill_row(row: str, level: dict[str, int]) -> str:
    cells = re.findall(r"<w:tc>.*?</w:tc>", row, flags=re.S)
    name = re.sub(r"<[^>]+>", "", cells[0]).strip().rstrip(".")
    if name not in level:
        return row
    cell = cells[1 + level[name]]
    assert not re.search(r"<w:t[ >]", cell), name
    new = cell.replace(SPACING, SPACING + '<w:jc w:val="center"/>', 1)
    new = new.replace("</w:p>", MARK + "</w:p>", 1)
    return row.replace(cell, new, 1)


def main() -> None:
    tex = TEX.read_text()
    level = levels(tex)
    body = tex.split(r"\section{Reflexión}")[1].split("\n", 2)[2]
    paras = [p for p in body.strip().split("\n\n") if p.strip()]

    doc = zipfile.ZipFile(SRC).read("word/document.xml").decode("utf8")
    i = doc.find("Empieza a escribir")
    ps, pe = doc.rfind("<w:p ", 0, i), doc.find("</w:p>", i) + len("</w:p>")
    doc = doc[:ps] + "".join(para_xml(p) for p in paras) + doc[pe:]
    doc = re.sub(r"<w:tr [^>]*>.*?</w:tr>", lambda m: fill_row(m.group(0), level), doc, flags=re.S)
    assert doc.count(">X</w:t>") == 17, doc.count(">X</w:t>")
    xml.dom.minidom.parseString(doc)

    with zipfile.ZipFile(SRC) as zin, zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = doc.encode("utf8") if item.filename == "word/document.xml" else zin.read(item.filename)
            zout.writestr(item, data)
    words = len(" ".join(t for p in paras for t, _ in runs(p)).split())
    print(f"{DST.relative_to(ROOT)}: 17 marks, {len(paras)} paragraphs, {words} words")


if __name__ == "__main__":
    main()
