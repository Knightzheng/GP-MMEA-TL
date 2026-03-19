param(
    [string]$TemplateDir = ".",
    [string]$ReportMarkdownPath = ".",
    [string]$OutputDir = ".",
    [string]$StudentName = "TO_FILL",
    [string]$StudentId = "TO_FILL",
    [string]$ClassName = "TO_FILL",
    [string]$Phone = "TO_FILL",
    [string]$Email = "TO_FILL",
    [string]$Internship = "TO_FILL",
    [string]$ThesisTitle = "TO_FILL",
    [string]$ReportDate = "TO_FILL",
    [string]$OtherText = "TO_FILL"
)

$ErrorActionPreference = "Stop"

function Get-SectionBodies {
    param([string]$Markdown)

    $parts = [regex]::Split($Markdown, '(?m)^## ')
    $bodies = @()
    foreach ($part in $parts) {
        if ([string]::IsNullOrWhiteSpace($part)) { continue }
        $normalized = $part -replace "`r`n", "`n"
        $firstBreak = $normalized.IndexOf("`n")
        if ($firstBreak -lt 0) { continue }
        $rest = $normalized.Substring($firstBreak + 1).TrimStart("`n")
        $secondBreak = $rest.IndexOf("`n`n")
        if ($secondBreak -ge 0) {
            $body = $rest.Substring($secondBreak + 2).Trim()
        } else {
            $body = $rest.Trim()
        }
        $bodies += $body
    }
    return ,$bodies
}

function Normalize-BodyText {
    param([string]$Text)

    $normalized = $Text -replace "`r`n", "`n"
    $paras = $normalized -split "`n`n+"
    $paras = $paras | ForEach-Object { ($_ -replace "`n", " ").Trim() } | Where-Object { $_ }
    return ($paras -join " ")
}

function Set-ParagraphText {
    param(
        $Document,
        [int]$ParagraphIndex,
        [string]$Text,
        [string]$FontName = "SimSun",
        [double]$FontSize = 12,
        [bool]$UseOnePointFive = $false,
        [bool]$Center = $false
    )

    $paragraph = $Document.Paragraphs.Item($ParagraphIndex)
    $range = $paragraph.Range
    $contentRange = $Document.Range($range.Start, $range.End - 1)
    $contentRange.Text = $Text
    $formattedRange = $Document.Range($contentRange.Start, $contentRange.End)
    $formattedRange.Font.NameFarEast = $FontName
    $formattedRange.Font.NameAscii = $FontName
    $formattedRange.Font.Size = $FontSize
    $formattedRange.ParagraphFormat.SpaceBefore = 0
    $formattedRange.ParagraphFormat.SpaceAfter = 0
    if ($UseOnePointFive) {
        $formattedRange.ParagraphFormat.LineSpacingRule = 1
    }
    if ($Center) {
        $formattedRange.ParagraphFormat.Alignment = 1
    }
}

function Insert-BodyAfterParagraph {
    param(
        $Document,
        [int]$AnchorParagraphIndex,
        [string]$Text,
        [string]$FontName = "SimSun",
        [double]$FontSize = 12,
        [bool]$UseOnePointFive = $true
    )

    $anchor = $Document.Paragraphs.Item($AnchorParagraphIndex)
    $anchor.Range.InsertParagraphAfter() | Out-Null
    $newParagraph = $Document.Paragraphs.Item($AnchorParagraphIndex + 1)
    $newParagraph.Range.InsertBefore($Text)
    $formattedRange = $newParagraph.Range
    $formattedRange.Font.NameFarEast = $FontName
    $formattedRange.Font.NameAscii = $FontName
    $formattedRange.Font.Size = $FontSize
    $formattedRange.ParagraphFormat.SpaceBefore = 0
    $formattedRange.ParagraphFormat.SpaceAfter = 0
    if ($UseOnePointFive) {
        $formattedRange.ParagraphFormat.LineSpacingRule = 1
    }
}

function Save-DocxAndPdf {
    param(
        $Document,
        [string]$BasePathWithoutExt
    )

    $docxPath = $BasePathWithoutExt + ".docx"
    $wdFormatXMLDocument = 16
    $Document.SaveAs([ref]$docxPath, [ref]$wdFormatXMLDocument)
}

if (-not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$templatePath = Get-ChildItem -LiteralPath $TemplateDir -Filter *.doc |
    Sort-Object Length -Descending |
    Select-Object -First 1 -ExpandProperty FullName

if (-not $templatePath) {
    throw "Could not find template .doc under $TemplateDir"
}

$markdown = Get-Content -LiteralPath $ReportMarkdownPath -Raw -Encoding UTF8
$sectionBodies = Get-SectionBodies -Markdown $markdown
if ($sectionBodies.Count -lt 5) {
    throw "Expected at least 5 markdown sections, got $($sectionBodies.Count)"
}

$summary = Normalize-BodyText -Text $sectionBodies[1]
$plan = Normalize-BodyText -Text $sectionBodies[2]
$problems = Normalize-BodyText -Text $sectionBodies[3]
$teacherComment = Normalize-BodyText -Text $sectionBodies[4]

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

try {
    $resolvedTemplate = (Resolve-Path -LiteralPath $templatePath).Path

    $docStudent = $word.Documents.Open($resolvedTemplate)
    Insert-BodyAfterParagraph -Document $docStudent -AnchorParagraphIndex 31 -Text $OtherText
    Insert-BodyAfterParagraph -Document $docStudent -AnchorParagraphIndex 29 -Text $problems
    Insert-BodyAfterParagraph -Document $docStudent -AnchorParagraphIndex 26 -Text $plan
    Insert-BodyAfterParagraph -Document $docStudent -AnchorParagraphIndex 22 -Text $summary
    Set-ParagraphText -Document $docStudent -ParagraphIndex 17 -Text ("设计（论文）题目    " + $ThesisTitle)
    Set-ParagraphText -Document $docStudent -ParagraphIndex 15 -Text ("实习单位    " + $Internship)
    Set-ParagraphText -Document $docStudent -ParagraphIndex 12 -Text ("电子邮箱    " + $Email)
    Set-ParagraphText -Document $docStudent -ParagraphIndex 10 -Text ("联系电话    " + $Phone)
    Set-ParagraphText -Document $docStudent -ParagraphIndex 7 -Text ("班  级    " + $ClassName)
    Set-ParagraphText -Document $docStudent -ParagraphIndex 5 -Text ("姓  名    " + $StudentName)
    Set-ParagraphText -Document $docStudent -ParagraphIndex 3 -Text ("学  号    " + $StudentId)
    Set-ParagraphText -Document $docStudent -ParagraphIndex 2 -Text $ReportDate -FontSize 10.5 -Center $true
    Save-DocxAndPdf -Document $docStudent -BasePathWithoutExt (Join-Path $OutputDir "midterm_report_student_20260319")
    $docStudent.Close([ref]0)

    $docTeacher = $word.Documents.Open($resolvedTemplate)
    Insert-BodyAfterParagraph -Document $docTeacher -AnchorParagraphIndex 34 -Text $teacherComment
    Insert-BodyAfterParagraph -Document $docTeacher -AnchorParagraphIndex 31 -Text $OtherText
    Insert-BodyAfterParagraph -Document $docTeacher -AnchorParagraphIndex 29 -Text $problems
    Insert-BodyAfterParagraph -Document $docTeacher -AnchorParagraphIndex 26 -Text $plan
    Insert-BodyAfterParagraph -Document $docTeacher -AnchorParagraphIndex 22 -Text $summary
    Set-ParagraphText -Document $docTeacher -ParagraphIndex 17 -Text ("设计（论文）题目    " + $ThesisTitle)
    Set-ParagraphText -Document $docTeacher -ParagraphIndex 15 -Text ("实习单位    " + $Internship)
    Set-ParagraphText -Document $docTeacher -ParagraphIndex 12 -Text ("电子邮箱    " + $Email)
    Set-ParagraphText -Document $docTeacher -ParagraphIndex 10 -Text ("联系电话    " + $Phone)
    Set-ParagraphText -Document $docTeacher -ParagraphIndex 7 -Text ("班  级    " + $ClassName)
    Set-ParagraphText -Document $docTeacher -ParagraphIndex 5 -Text ("姓  名    " + $StudentName)
    Set-ParagraphText -Document $docTeacher -ParagraphIndex 3 -Text ("学  号    " + $StudentId)
    Set-ParagraphText -Document $docTeacher -ParagraphIndex 2 -Text $ReportDate -FontSize 10.5 -Center $true
    Save-DocxAndPdf -Document $docTeacher -BasePathWithoutExt (Join-Path $OutputDir "midterm_report_teacher_ref_20260319")
    $docTeacher.Close([ref]0)
}
finally {
    $word.Quit()
}
