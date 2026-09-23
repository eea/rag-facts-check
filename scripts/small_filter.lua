-- Pandoc Lua filter to preserve <small> tags across PPTX output
function RawInline(elem)
  if elem.format == 'html' and elem.text == '<small>' then
    return pandoc.Str('⟦SMALL⟧')
  elseif elem.format == 'html' and elem.text == '</small>' then
    return pandoc.Str('⟦/SMALL⟧')
  end
end
