import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Canvas class to dynamically compute and print total page count on footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        # Suppress footer on cover page
        if self._pageNumber == 1:
            return
            
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#4A5568"))
        
        # Header (Top of Page)
        self.drawString(inch, 10.5 * inch, "Physics RAG Chatbot — Evaluation & Hallucination Report")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(inch, 10.4 * inch, 7.5 * inch, 10.4 * inch)
        
        # Footer (Bottom of Page)
        self.line(inch, 0.75 * inch, 7.5 * inch, 0.75 * inch)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(7.5 * inch, 0.55 * inch, page_text)
        self.drawString(inch, 0.55 * inch, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.restoreState()


def build_pdf_report(json_path, pdf_path):
    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    metrics = data["metrics"]
    results = data["detailed_results"]
    
    # Setup document
    margin = 0.5 * inch
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=0.75 * inch,
        bottomMargin=0.85 * inch
    )
    
    # Setup styles
    styles = getSampleStyleSheet()
    
    # Custom Styles
    style_title = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=34,
        textColor=colors.HexColor("#1E293B"),
        alignment=0, # Left-aligned
        spaceAfter=12
    )
    
    style_subtitle = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=25
    )
    
    style_h1 = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=15,
        spaceAfter=8,
        keepWithNext=True
    )
    
    style_h2 = ParagraphStyle(
        'SubSectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#334155"),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    style_body = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155")
    )
    
    style_code = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A")
    )

    style_metric_label = ParagraphStyle(
        'MetricLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        textColor=colors.white
    )
    
    style_table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1E293B")
    )
    
    style_table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=style_table_cell,
        fontName='Helvetica-Bold'
    )
    
    story = []
    
    # ── COVER PAGE ──────────────────────────────────────────────────────────
    # Large colored accent block
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("Local Physics RAG Chatbot", style_subtitle))
    story.append(Paragraph("System Performance & Evaluation Report", style_title))
    
    # Meta Info block
    meta_data = [
        [Paragraph("<b>Evaluation Date:</b>", style_table_cell), Paragraph(datetime.now().strftime("%B %d, %Y"), style_table_cell)],
        [Paragraph("<b>Underlying LLM Model:</b>", style_table_cell), Paragraph(data.get("model", "llama-3.1-8b-instant"), style_table_cell)],
        [Paragraph("<b>Embedding Model:</b>", style_table_cell), Paragraph("sentence-transformers/all-mpnet-base-v2 (GPU)", style_table_cell)],
        [Paragraph("<b>Reranker Model:</b>", style_table_cell), Paragraph("ms-marco-MiniLM-L-6-v2", style_table_cell)],
        [Paragraph("<b>Total Questions Tested:</b>", style_table_cell), Paragraph(str(len(results)), style_table_cell)]
    ]
    t_meta = Table(meta_data, colWidths=[2.2 * inch, 4.8 * inch])
    t_meta.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 0.4 * inch))
    
    # ── METRICS SECTION ──────────────────────────────────────────────────────
    story.append(Paragraph("Performance Metrics Dashboard", style_h1))
    
    def format_pct(val):
        return f"{val * 100:.2f}%" if val <= 1.0 else f"{val:.2f}%"
        
    metrics_data = [
        [
            Paragraph("Metric Description", style_metric_label),
            Paragraph("Target Requirement", style_metric_label),
            Paragraph("System Score", style_metric_label),
            Paragraph("Status", style_metric_label)
        ],
        [
            Paragraph("<b>Citation Accuracy</b><br/><font color='#64748B' size=8>Correctly formatted textbook citations matching text</font>", style_table_cell),
            Paragraph("&ge; 85.00%", style_table_cell),
            Paragraph(format_pct(metrics["citation_accuracy"]), style_table_cell_bold),
            Paragraph("<font color='green'><b>PASSED</b></font>" if metrics["citation_accuracy"] >= 0.85 else "<font color='red'><b>FAILED</b></font>", style_table_cell_bold)
        ],
        [
            Paragraph("<b>Out-of-Scope (OOS) Refusal Rate</b><br/><font color='#64748B' size=8>Politely rejecting non-physics queries without calling LLM</font>", style_table_cell),
            Paragraph("&ge; 90.00%", style_table_cell),
            Paragraph(format_pct(metrics["oos_refusal_rate"]), style_table_cell_bold),
            Paragraph("<font color='green'><b>PASSED</b></font>" if metrics["oos_refusal_rate"] >= 0.90 else "<font color='red'><b>FAILED</b></font>", style_table_cell_bold)
        ],
        [
            Paragraph("<b>Edge Case Pass Rate</b><br/><font color='#64748B' size=8>Physics-adjacent terms that should NOT be blocked</font>", style_table_cell),
            Paragraph("100.00%", style_table_cell),
            Paragraph(format_pct(metrics["edge_case_pass_rate"]), style_table_cell_bold),
            Paragraph("<font color='green'><b>PASSED</b></font>" if metrics["edge_case_pass_rate"] >= 1.0 else "<font color='red'><b>FAILED</b></font>", style_table_cell_bold)
        ],
        [
            Paragraph("<b>Hallucination Rate</b><br/><font color='#64748B' size=8>Textbook keywords generated by LLM absent in retrieved context</font>", style_table_cell),
            Paragraph("&lt; 10.00%", style_table_cell),
            Paragraph(format_pct(metrics["hallucination_rate"]), style_table_cell_bold),
            Paragraph("<font color='orange'><b>MONITOR</b></font>" if metrics["hallucination_rate"] >= 0.10 else "<font color='green'><b>PASSED</b></font>", style_table_cell_bold)
        ],
        [
            Paragraph("<b>Retrieval Precision@5</b><br/><font color='#64748B' size=8>Ratio of retrieved chunks containing relevant terms</font>", style_table_cell),
            Paragraph("Informational", style_table_cell),
            Paragraph(f"{metrics['avg_precision']:.3f}", style_table_cell),
            Paragraph("N/A", style_table_cell)
        ],
        [
            Paragraph("<b>Retrieval Recall@5</b><br/><font color='#64748B' size=8>Coverage of keywords compared to top-50 candidate pool</font>", style_table_cell),
            Paragraph("Informational", style_table_cell),
            Paragraph(f"{metrics['avg_recall']:.3f}", style_table_cell),
            Paragraph("N/A", style_table_cell)
        ],
        [
            Paragraph("<b>Avg Retrieval Strength Score</b><br/><font color='#64748B' size=8>Mean reranked cosine similarity of top passages</font>", style_table_cell),
            Paragraph("Informational", style_table_cell),
            Paragraph(f"{metrics['avg_retrieval_strength']:.3f}", style_table_cell),
            Paragraph("N/A", style_table_cell)
        ],
    ]
    
    t_metrics = Table(metrics_data, colWidths=[3.2 * inch, 1.4 * inch, 1.2 * inch, 1.2 * inch])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_metrics)
    
    story.append(PageBreak())
    
    # ── DETAILED RESULTS SECTION ─────────────────────────────────────────────
    story.append(Paragraph("Detailed Question & Answer Logs", style_h1))
    
    for idx, q_res in enumerate(results):
        q_elements = []
        q_elements.append(Spacer(1, 10))
        
        # ID and Category Header
        header_text = f"<b>[{q_res['id']}] Category: {q_res['category']}</b> (Type: {q_res['type'].upper()})"
        q_elements.append(Paragraph(header_text, style_h2))
        
        # Question Block
        q_elements.append(Paragraph(f"<b>Question:</b> {q_res['question']}", style_body))
        q_elements.append(Spacer(1, 4))
        
        # Answer Block
        ans_text = q_res['answer'].replace("\n", "<br/>")
        q_elements.append(Paragraph(f"<b>Answer:</b> {ans_text}", style_body))
        q_elements.append(Spacer(1, 4))
        
        # Metadata Table
        meta_rows = []
        
        if q_res["type"] == "physics":
            citations = ", ".join(q_res.get("citations_returned", [])) or "None"
            meta_rows.append([
                Paragraph("<b>Citations:</b>", style_table_cell),
                Paragraph(citations, style_table_cell)
            ])
            
            status_labels = []
            if q_res.get("has_valid_citations"):
                status_labels.append("<font color='green'>Valid Citations</font>")
            else:
                status_labels.append("<font color='red'>Missing Citations</font>")
                
            if q_res.get("is_hallucinated"):
                status_labels.append("<font color='red'>Potential Hallucination</font>")
            else:
                status_labels.append("<font color='green'>Factual (No Hallucination)</font>")
                
            meta_rows.append([
                Paragraph("<b>Validation Status:</b>", style_table_cell),
                Paragraph(" | ".join(status_labels), style_table_cell)
            ])
            
            retrieval_meta = f"Strength: {q_res.get('retrieval_strength', 0.0):.3f} ({q_res.get('retrieval_strength_label', 'N/A')}) | Precision: {q_res.get('precision', 0.0):.2f} | Recall: {q_res.get('recall', 0.0):.2f}"
            meta_rows.append([
                Paragraph("<b>Retrieval Metrics:</b>", style_table_cell),
                Paragraph(retrieval_meta, style_table_cell)
            ])
            
        elif q_res["type"] == "oos":
            is_ok = q_res.get("refused_correctly")
            refusal_status = "<font color='green'><b>Correctly Refused</b></font>" if is_ok else "<font color='red'><b>Failed Refusal (Let Through)</b></font>"
            meta_rows.append([
                Paragraph("<b>Refusal Check:</b>", style_table_cell),
                Paragraph(refusal_status, style_table_cell)
            ])
            
        elif q_res["type"] == "edge":
            is_ok = q_res.get("passed_correctly")
            pass_status = "<font color='green'><b>Passed Correctly (Allowed)</b></font>" if is_ok else "<font color='red'><b>Failed (Erroneously Blocked)</b></font>"
            meta_rows.append([
                Paragraph("<b>Pass Check:</b>", style_table_cell),
                Paragraph(pass_status, style_table_cell)
            ])
            
        t_meta_q = Table(meta_rows, colWidths=[1.8 * inch, 5.2 * inch])
        t_meta_q.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
            ('LINELEFT', (0,0), (0,-1), 3, colors.HexColor("#3B82F6") if q_res["type"] == "physics" else (colors.HexColor("#10B981") if q_res["type"] == "edge" else colors.HexColor("#EF4444"))),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ]))
        
        q_elements.append(t_meta_q)
        q_elements.append(Spacer(1, 10))
        
        # Keep each Q&A block intact on a single page if possible
        story.append(KeepTogether(q_elements))
        
    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF report successfully compiled at {pdf_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_in = os.path.join(base_dir, "tests", "results", "report_20260611.json")
    pdf_out = os.path.join(base_dir, "tests", "results", "report_20260611.pdf")
    
    if os.path.exists(json_in):
        build_pdf_report(json_in, pdf_out)
    else:
        print(f"ERROR: {json_in} not found.")
