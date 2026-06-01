-- table_vlines.lua
-- Replaces all pandoc tables with centered LaTeX tabular:
--   • natural-width l/r/c columns (not \linewidth-based)
--   • vertical column lines
--   • horizontal rules after every row
--   • centered on the page

local function inlines_to_latex(inlines)
  -- Render a list of inlines to a flat LaTeX string via pandoc's writer
  local s = pandoc.write(pandoc.Pandoc({pandoc.Para(inlines)}), "latex")
  return s:gsub("%s*\n%s*", " "):gsub("^%s+", ""):gsub("%s+$", "")
end

function Para(el)
  -- 1. Center captions starting with "Table" or "Equation"
  local text = pandoc.utils.stringify(el.content)
  if text:match("^Table %d+%.") or text:match("^Equation %d+%.") then
    return pandoc.RawBlock("latex",
      "\\begin{center}" .. text .. "\\end{center}")
  end

  -- 2. Hanging indent for reference paragraphs starting with [N]
  local first = el.content[1]
  if first and first.t == "Str" and first.text:match("^%[%d+%]$") then
    local label = first.text  -- e.g. "[1]"
    -- Drop the label and the space that follows it
    local rest = {}
    for i = 2, #el.content do
      if i == 2 and el.content[i].t == "Space" then
        -- skip the space immediately after the label
      else
        table.insert(rest, el.content[i])
      end
    end
    local body = inlines_to_latex(rest)
    -- \makebox[2em][l] gives every label the same fixed width,
    -- \hangindent=2em aligns all wrapped lines under the text start
    return pandoc.RawBlock("latex",
      "\\noindent\\hangindent=2em\\hangafter=1" ..
      "\\makebox[2em][l]{" .. label .. "}" .. body .. "\n\n")
  end

  return el
end

local function align_char(align)
  local t = align.t
  if     t == "AlignRight"  then return "r"
  elseif t == "AlignCenter" then return "c"
  else                           return "l"
  end
end

local function cell_to_latex(cell)
  -- Convert cell content to a flat LaTeX string via pandoc's own writer
  local doc = pandoc.Pandoc(cell.contents)
  local s   = pandoc.write(doc, "latex")
  -- Collapse newlines / leading-trailing whitespace
  s = s:gsub("%s*\n%s*", " "):gsub("^%s+", ""):gsub("%s+$", "")
  return s
end

function Table(el)
  -- Build column spec: |l|l|l| etc.
  local cols = {}
  for _, cs in ipairs(el.colspecs) do
    table.insert(cols, align_char(cs[1]))
  end
  local colspec = "|" .. table.concat(cols, "|") .. "|"

  local out = {}
  table.insert(out, "\\begin{center}")
  table.insert(out, "\\begin{tabular}{" .. colspec .. "}")
  table.insert(out, "\\hline")

  -- Header row (bold)
  if el.head and el.head.rows and #el.head.rows > 0 then
    local hcells = {}
    for _, cell in ipairs(el.head.rows[1].cells) do
      table.insert(hcells, "\\textbf{" .. cell_to_latex(cell) .. "}")
    end
    table.insert(out, table.concat(hcells, " & ") .. " \\\\")
    table.insert(out, "\\hline")
  end

  -- Body rows
  for _, tbody in ipairs(el.bodies) do
    for _, row in ipairs(tbody.body) do
      local bcells = {}
      for _, cell in ipairs(row.cells) do
        table.insert(bcells, cell_to_latex(cell))
      end
      table.insert(out, table.concat(bcells, " & ") .. " \\\\")
      table.insert(out, "\\hline")
    end
  end

  table.insert(out, "\\end{tabular}")
  table.insert(out, "\\end{center}")

  return pandoc.RawBlock("latex", table.concat(out, "\n"))
end
