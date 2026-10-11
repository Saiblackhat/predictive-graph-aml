"""
Suspicious Activity Report (SAR) PDF Generator (Sprint 4).

Generates compliance-grade FinCEN-style SAR documents in PDF format using ReportLab.
Contains:
  - Header with Sample/Synthetic Notice
  - Account Subject Details & Risk Tiers
  - Primary Detected Money Laundering Typology
  - Top 5 SHAP Contributing Explanations in Plain English
  - Transaction History Table
  - Analyst Investigation Notes & Audit History
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_sar_pdf(case_data: Dict[str, Any]) -> bytes:
    """
    Generate SAR PDF report as in-memory bytes buffer.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1E293B"),
    )

    banner_style = ParagraphStyle(
        "Banner",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#DC2626"),
        alignment=1,  # Center
    )

    section_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )

    elements = []

    # 1. Prominent Synthetic Disclaimer
    elements.append(
        Paragraph(
            "*** SAMPLE DOCUMENT GENERATED FROM SYNTHETIC AML DATASET - FOR COMPLIANCE DEMO ONLY ***",
            banner_style,
        )
    )
    elements.append(Spacer(1, 10))

    # 2. Header
    elements.append(Paragraph("FINANCIAL CRIMES ENFORCEMENT NETWORK (FinCEN) — SAR DOSSIER", title_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0F172A"), spaceBefore=4, spaceAfter=12))

    # 3. Case & Subject Information Table
    case_id = case_data.get("case_id", "N/A")
    acc_id = case_data.get("account_id", "N/A")
    score = case_data.get("risk_score", 0.0)
    tier = case_data.get("risk_tier", "CRITICAL")
    typology = case_data.get("typology", "SUSPICIOUS")
    status = case_data.get("status", "OPEN")
    created_at = case_data.get("created_at", "2026-10-11")

    info_data = [
        [
            Paragraph("<b>Case ID:</b>", body_style),
            Paragraph(f"CASE-{case_id}", body_style),
            Paragraph("<b>Status:</b>", body_style),
            Paragraph(str(status), body_style),
        ],
        [
            Paragraph("<b>Subject Account:</b>", body_style),
            Paragraph(str(acc_id), body_style),
            Paragraph("<b>Date Opened:</b>", body_style),
            Paragraph(str(created_at), body_style),
        ],
        [
            Paragraph("<b>AML Risk Score:</b>", body_style),
            Paragraph(f"<b>{score:.1f} / 100</b> ({tier})", body_style),
            Paragraph("<b>Primary Typology:</b>", body_style),
            Paragraph(str(typology), body_style),
        ],
    ]

    t_info = Table(info_data, colWidths=[100, 170, 100, 170])
    t_info.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    elements.append(t_info)
    elements.append(Spacer(1, 12))

    # 4. Top SHAP Risk Explanations
    elements.append(Paragraph("1. Explainable Risk Factors (SHAP Model Drivers)", section_style))
    shap_reasons = case_data.get("shap_reasons", [])
    if shap_reasons:
        shap_table_data = [[Paragraph("<b>Factor & Topological Pattern</b>", body_style), Paragraph("<b>Risk Impact</b>", body_style)]]
        for r in shap_reasons:
            plain_txt = r.get("plain_language", r.get("feature", ""))
            impact_txt = r.get("impact", "RAISES_RISK")
            shap_table_data.append([
                Paragraph(plain_txt, body_style),
                Paragraph(f"<font color='red'><b>{impact_txt}</b></font>", body_style)
            ])
        t_shap = Table(shap_table_data, colWidths=[420, 120])
        t_shap.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        elements.append(t_shap)
    else:
        elements.append(Paragraph("No SHAP factors recorded.", body_style))
    elements.append(Spacer(1, 12))

    # 5. Transaction Sample Table
    elements.append(Paragraph("2. Recent Flagged Transactions", section_style))
    txns = case_data.get("transactions", [])
    if txns:
        txn_table_data = [
            [
                Paragraph("<b>Date</b>", body_style),
                Paragraph("<b>Direction</b>", body_style),
                Paragraph("<b>Counterparty</b>", body_style),
                Paragraph("<b>Amount (USD)</b>", body_style),
                Paragraph("<b>Type</b>", body_style),
            ]
        ]
        for tx in txns[:6]:
            txn_table_data.append(
                [
                    Paragraph(str(tx.get("timestamp", ""))[:19], body_style),
                    Paragraph(str(tx.get("direction", "OUT")), body_style),
                    Paragraph(str(tx.get("counterparty", "")), body_style),
                    Paragraph(f"${float(tx.get('amount', 0.0)):,.2f}", body_style),
                    Paragraph(str(tx.get("transaction_type", "TRANSFER")), body_style),
                ]
            )
        t_tx = Table(txn_table_data, colWidths=[110, 60, 140, 110, 120])
        t_tx.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        elements.append(t_tx)
    else:
        elements.append(Paragraph("No individual transactions loaded.", body_style))
    elements.append(Spacer(1, 12))

    # 6. Analyst Notes & Certification
    elements.append(Paragraph("3. Compliance Analyst Assessment & Notes", section_style))
    notes = case_data.get("notes", [])
    if notes:
        for n in notes:
            elements.append(Paragraph(f"• <b>{n.get('author', 'Analyst')}</b> ({str(n.get('date', ''))[:19]}): {n.get('text', '')}", body_style))
            elements.append(Spacer(1, 3))
    else:
        elements.append(Paragraph("<i>Initial automated referral. Pending formal analyst review.</i>", body_style))

    elements.append(Spacer(1, 16))
    elements.append(
        Paragraph(
            "<b>Compliance Officer Signature:</b> ___________________________   <b>Date:</b> _____________",
            body_style,
        )
    )

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
