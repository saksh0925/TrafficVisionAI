import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.units import cm
import io
import os


def create_evidence_image(img_bgr, detections, timestamp=None):
    """
    Creates a final annotated evidence image with a dark header bar.
    Returns PIL Image.
    """
    from detector import draw_detections
    annotated = draw_detections(img_bgr, detections)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    violations = [d for d in detections if d["is_violation"]]
    header_h = 60
    h, w = annotated.shape[:2]

    canvas = np.zeros((h + header_h, w, 3), dtype=np.uint8)
    canvas[:header_h] = (30, 30, 30)
    canvas[header_h:] = annotated

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "TRAFFIC VIOLATION EVIDENCE", (12, 24),
                font, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Timestamp: {timestamp}", (12, 44),
                font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Violations detected: {len(violations)}", (w - 240, 24),
                font, 0.55, (0, 200, 100) if len(violations) == 0 else (0, 80, 255), 1, cv2.LINE_AA)

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def create_pdf_report(detections, evidence_pil_image, case_id=None):
    """
    Generates a PDF violation report.
    Returns bytes of the PDF.
    """
    if case_id is None:
        case_id = datetime.now().strftime("TRF-%Y%m%d-%H%M%S")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    violations = [d for d in detections if d["is_violation"]]

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                 fontSize=18, spaceAfter=6, textColor=colors.HexColor('#1a1a2e'))
    sub_style = ParagraphStyle('Sub', parent=styles['Normal'],
                               fontSize=10, textColor=colors.grey, spaceAfter=16)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
                                   fontSize=12, spaceBefore=14, spaceAfter=6,
                                   textColor=colors.HexColor('#185FA5'))
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, spaceAfter=6)

    story = []

    story.append(Paragraph("Traffic Violation Evidence Report", title_style))
    story.append(Paragraph(f"Case ID: {case_id} | Generated: {timestamp}", sub_style))
    story.append(Spacer(1, 0.3*cm))

    # Evidence image
    img_buffer = io.BytesIO()
    evidence_pil_image.save(img_buffer, format='JPEG', quality=85)
    img_buffer.seek(0)
    rl_img = RLImage(img_buffer, width=16*cm, height=10*cm)
    story.append(rl_img)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Summary", heading_style))
    summary_data = [
        ["Total detections", str(len(detections))],
        ["Violations found", str(len(violations))],
        ["Timestamp", timestamp],
        ["Case ID", case_id],
    ]
    summary_table = Table(summary_data, colWidths=[6*cm, 10*cm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E6F1FB')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#185FA5')),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (1, 0), (1, -1), [colors.white, colors.HexColor('#f8f8f8')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.4*cm))

    if violations:
        story.append(Paragraph("Violation Details", heading_style))
        viol_data = [["#", "Violation Type", "Confidence", "Plate No."]]
        for i, v in enumerate(violations, 1):
            plate = v.get("plate_text") or "N/A"
            viol_data.append([
                str(i),
                v["label"],
                f"{v['confidence']:.1%}",
                plate,
            ])

        viol_table = Table(viol_data, colWidths=[1*cm, 8*cm, 3.5*cm, 4*cm])
        viol_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#185FA5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(viol_table)
    else:
        story.append(Paragraph("No violations detected in this image.", body_style))

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "This report is auto-generated by TrafficVision AI. "
        "Confidence scores above 40% are flagged for review.",
        ParagraphStyle('Footer', parent=styles['Normal'],
                       fontSize=8, textColor=colors.grey)))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
