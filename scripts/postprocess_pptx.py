"""Post-process Pandoc-generated PPTX files to apply font size styling for <small> tags."""

import os
import re
import sys
import zipfile


def process_slide_xml(xml_content: str) -> str:
    """Find text runs containing small markers and wrap them with sz="1300" (13pt)."""

    def replace_run(match):
        full_run = match.group(0)
        if "⟦SMALL⟧" not in full_run:
            return full_run

        t_match = re.search(r"<a:t[^>]*>(.*?)</a:t>", full_run, re.DOTALL)
        if not t_match:
            return full_run

        inner_text = t_match.group(1)
        m = re.search(r"(.*?)⟦SMALL⟧(.*?)⟦/SMALL⟧(.*)", inner_text, re.DOTALL)
        if not m:
            return full_run

        before, small, after = m.group(1), m.group(2), m.group(3)
        res = []
        if before:
            res.append(f'<a:r><a:t xml:space="preserve">{before}</a:t></a:r>')
        res.append(
            f'<a:r><a:rPr sz="1300"/><a:t xml:space="preserve"> {small.strip()}</a:t></a:r>'
        )
        if after:
            res.append(f'<a:r><a:t xml:space="preserve">{after}</a:t></a:r>')
        return "".join(res)

    return re.sub(
        r"<a:r>(?:<a:rPr[^>]*/>|<a:rPr>.*?</a:rPr>)?<a:t[^>]*>.*?</a:t></a:r>",
        replace_run,
        xml_content,
        flags=re.DOTALL,
    )


def postprocess_pptx(filepath: str) -> None:
    temp_path = filepath + ".tmp"
    with (
        zipfile.ZipFile(filepath, "r") as zin,
        zipfile.ZipFile(temp_path, "w") as zout,
    ):
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.startswith("ppt/slides/slide") and item.filename.endswith(".xml"):
                text = data.decode("utf-8")
                new_text = process_slide_xml(text)
                data = new_text.encode("utf-8")
            zout.writestr(item, data)
    os.replace(temp_path, filepath)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        if os.path.exists(path):
            postprocess_pptx(path)
