"use client";

/**
 * PDF Generator utility for multilingual content
 * Simple direct text rendering approach to avoid CSS issues
 */

import { jsPDF } from "jspdf";
import "jspdf-autotable";

/**
 * Format date for PDF
 * @param {Date|string} dateString - Date to format
 * @param {string} locale - Locale for date formatting
 * @returns {string} Formatted date string
 */
export function formatPdfDate(dateString, locale = "en") {
  return new Date(dateString).toLocaleString(locale, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * Generate PDF with batch descriptions
 * @param {Object|Array} batchInput - Batch object or array of batch objects with descriptions
 * @param {Object} languageDisplay - Map of language codes to display names
 * @param {string} locale - Current locale for date formatting
 * @returns {Promise<boolean>} - True if successful, false if error
 */
export async function generateAnalysisPDF(
  batchInput,
  languageDisplay,
  locale = "en"
) {
  try {
    // Handle array input - use the first batch
    const batch = Array.isArray(batchInput) ? batchInput[0] : batchInput;

    // If no batch, return early
    if (!batch) {
      console.error("No valid batch provided");
      return false;
    }

    // Create a new PDF document
    const doc = new jsPDF();

    // Set initial position
    let yPosition = 20;
    const pageHeight = doc.internal.pageSize.height;
    const margin = 20;
    const lineHeight = 10;

    // Get window location for web link
    const baseUrl =
      typeof window !== "undefined"
        ? `${window.location.protocol}//${window.location.host}`
        : "";
    const batchUrl = `${baseUrl}/${locale}/dashboard/batches/${batch.id}`;

    // Function to generate QR code - only if needed for non-Latin scripts
    const hasNonLatinDescriptions =
      batch.descriptions &&
      batch.descriptions.some((desc) => desc.language !== "En");

    // Title
    doc.setFontSize(20);
    doc.setFont("helvetica", "bold");
    doc.text(
      `Crop Analysis Report: ${batch.name || "Untitled"}`,
      margin,
      yPosition
    );
    yPosition += 20;

    // Batch Info
    doc.setFontSize(12);
    doc.setFont("helvetica", "normal");
    doc.text(`Crop Type: ${batch.cropType || "Unknown"}`, margin, yPosition);
    yPosition += lineHeight;
    doc.text(`Images Count: ${batch.imagesCount || 0}`, margin, yPosition);
    yPosition += lineHeight;
    doc.text(
      `Created: ${formatPdfDate(batch.createdAt, locale)}`,
      margin,
      yPosition
    );
    yPosition += lineHeight * 2;

    // Process descriptions
    if (batch.descriptions && batch.descriptions.length > 0) {
      // For English descriptions - fully supported
      const enDesc = batch.descriptions.find((d) => d.language === "En");
      if (enDesc) {
        // English heading
        doc.setFontSize(16);
        doc.setFont("helvetica", "bold");
        doc.text("English Analysis", margin, yPosition);
        yPosition += lineHeight * 1.5;

        // Long description
        if (enDesc.longDescription) {
          doc.setFontSize(14);
          doc.setFont("helvetica", "bold");
          doc.text("Briefed Analysis:", margin, yPosition);
          yPosition += lineHeight * 1.5;

          doc.setFontSize(10);
          doc.setFont("helvetica", "normal");

          const longDesc = enDesc.longDescription || "No description available";
          const briefedLines = doc.splitTextToSize(longDesc, 170);

          briefedLines.forEach((line) => {
            if (yPosition > pageHeight - 20) {
              doc.addPage();
              yPosition = margin;
            }
            doc.text(line, margin, yPosition);
            yPosition += lineHeight * 0.8;
          });

          yPosition += lineHeight;
        }

        // Short description
        if (enDesc.shortDescription) {
          doc.setFontSize(14);
          doc.setFont("helvetica", "bold");
          doc.text("Summarised Analysis:", margin, yPosition);
          yPosition += lineHeight * 1.5;

          doc.setFontSize(10);
          doc.setFont("helvetica", "normal");

          const summaryLines = doc.splitTextToSize(
            enDesc.shortDescription,
            170
          );

          summaryLines.forEach((line) => {
            if (yPosition > pageHeight - 20) {
              doc.addPage();
              yPosition = margin;
            }
            doc.text(line, margin, yPosition);
            yPosition += lineHeight * 0.8;
          });
        }

        yPosition += lineHeight * 2;
      }

      // For non-English descriptions
      const nonEnglishDescs = batch.descriptions.filter(
        (d) => d.language !== "En"
      );

      if (nonEnglishDescs.length > 0) {
        // Add header for other languages
        if (yPosition > pageHeight - 60) {
          doc.addPage();
          yPosition = margin;
        }

        doc.setFontSize(14);
        doc.setFont("helvetica", "bold");
        doc.text("Other Languages", margin, yPosition);
        yPosition += lineHeight * 1.5;

        // List all available non-English languages
        doc.setFontSize(10);
        doc.setFont("helvetica", "normal");
        doc.text(
          "This PDF includes descriptions in the following languages:",
          margin,
          yPosition
        );
        yPosition += lineHeight * 1.5;

        // List languages
        nonEnglishDescs.forEach((desc) => {
          const langName = languageDisplay[desc.language] || desc.language;
          doc.text(`• ${langName}`, margin + 5, yPosition);
          yPosition += lineHeight;
        });

        yPosition += lineHeight * 2;

        // Important notice
        doc.setFillColor(245, 245, 245);
        doc.rect(margin, yPosition, 170, 25, "F");

        doc.setFontSize(12);
        doc.setFont("helvetica", "bold");
        doc.text("Important Note:", margin + 5, yPosition + 8);

        doc.setFontSize(10);
        doc.setFont("helvetica", "normal");
        doc.text(
          "Non-Latin scripts (like Kannada) cannot be displayed correctly in this PDF.",
          margin + 5,
          yPosition + 16
        );
        doc.text(
          "Please use the web interface to view all languages properly.",
          margin + 5,
          yPosition + 24
        );

        yPosition += 35;
      }
    } else if (batch.description) {
      // Legacy description support
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.text("Description:", margin, yPosition);
      yPosition += lineHeight * 1.5;

      doc.setFontSize(10);
      doc.setFont("helvetica", "normal");

      const briefedLines = doc.splitTextToSize(batch.description, 170);

      briefedLines.forEach((line) => {
        if (yPosition > pageHeight - 20) {
          doc.addPage();
          yPosition = margin;
        }
        doc.text(line, margin, yPosition);
        yPosition += lineHeight * 0.8;
      });
    }

    // Add a footer with link back to web interface
    doc.addPage();
    yPosition = 40;

    doc.setFontSize(16);
    doc.setFont("helvetica", "bold");
    doc.text("Access Full Multilingual Content", margin, yPosition);
    yPosition += lineHeight * 2;

    // Create a colored box for the link
    doc.setFillColor(230, 245, 230);
    doc.rect(margin, yPosition, 170, 30, "F");

    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    doc.text("View Online:", margin + 5, yPosition + 12);

    // Show the URL
    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.text(`${batchUrl}`, margin + 40, yPosition + 12);

    doc.setFontSize(10);
    doc.setFont("helvetica", "italic");
    doc.text(
      "(Copy and paste this URL into your browser to access all languages)",
      margin + 5,
      yPosition + 22
    );

    yPosition += 40;

    // Add info box
    doc.setFillColor(240, 240, 240);
    doc.rect(margin, yPosition, 170, 40, "F");

    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    doc.text("About This Report", margin + 5, yPosition + 10);

    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.text(
      "This PDF provides English content and a list of other available languages.",
      margin + 5,
      yPosition + 20
    );
    doc.text(
      "Due to technical limitations, non-Latin scripts (like Kannada) cannot be displayed",
      margin + 5,
      yPosition + 30
    );
    doc.text(
      "correctly in this PDF. Please use the web interface for the complete experience.",
      margin + 5,
      yPosition + 40
    );

    // Save the PDF
    doc.save(`${batch.name}_analysis_report.pdf`);

    return true;
  } catch (error) {
    console.error("Error generating PDF:", error);
    return false;
  }
}

/**
 * For backward compatibility - redirects to generateAnalysisPDF
 */
export async function generateBatchReportFromState(
  batch,
  languageDisplay,
  locale = "en"
) {
  return generateAnalysisPDF(batch, languageDisplay, locale);
}

/**
 * For backward compatibility - not actively used
 */
export async function generateBatchReport(
  batchId,
  languageDisplay,
  locale = "en"
) {
  try {
    // 1. Fetch batch details
    const response = await fetch(`/api/dashboard/batches/${batchId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch batch: ${response.statusText}`);
    }

    const { batch } = await response.json();
    if (!batch) {
      throw new Error("Batch not found");
    }

    // Use the main PDF generation function
    return generateAnalysisPDF(batch, languageDisplay, locale);
  } catch (error) {
    console.error("Error in generateBatchReport:", error);
    return false;
  }
}

/**
 * Legacy compatibility function - redirects to main function
 */
export async function generatePDF(batch, languageDisplay, locale = "en") {
  console.warn(
    "Using legacy generatePDF function - consider updating your code"
  );
  return generateAnalysisPDF(batch, languageDisplay, locale);
}
