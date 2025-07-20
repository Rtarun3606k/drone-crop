"use client";

/**
 * PDF Generator utility for multilingual content
 * Using pdfmake with support for non-Latin scripts
 */

import pdfMake from "pdfmake/build/pdfmake";
import pdfFonts from "pdfmake/build/vfs_fonts";

// Initialize pdfmake
pdfMake.vfs = pdfFonts.pdfMake.vfs;

// Define fonts with fallbacks for better multilingual support
const fonts = {
  Roboto: {
    normal: "Roboto-Regular.ttf",
    bold: "Roboto-Medium.ttf",
    italics: "Roboto-Italic.ttf",
    bolditalics: "Roboto-MediumItalic.ttf",
  },
  // For older browsers that might not support all Unicode ranges
  fallback: {
    normal: "DroidSansFallback.ttf",
  },
};

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

    // Get window location for web link
    const baseUrl =
      typeof window !== "undefined"
        ? `${window.location.protocol}//${window.location.host}`
        : "";
    const batchUrl = `${baseUrl}/${locale}/dashboard/batches/${batch.id}`;

    // Start building the document definition
    const docDefinition = {
      pageSize: "A4",
      pageMargins: [40, 60, 40, 60],
      defaultStyle: {
        font: "Roboto",
      },
      content: [],
    };

    // Title
    docDefinition.content.push({
      text: `Crop Analysis Report: ${batch.name || "Untitled"}`,
      style: "header",
      margin: [0, 0, 0, 20],
    });

    // Batch Information
    docDefinition.content.push(
      { text: "Batch Information", style: "subheader" },
      {
        layout: "lightHorizontalLines",
        table: {
          headerRows: 0,
          widths: ["30%", "70%"],
          body: [
            ["Crop Type", batch.cropType || "Unknown"],
            ["Images Count", batch.imagesCount || 0],
            ["Created On", formatPdfDate(batch.createdAt, locale)],
            [
              "Preferred Language",
              languageDisplay[batch.preferredLanguage] ||
                batch.preferredLanguage,
            ],
          ],
        },
        margin: [0, 0, 0, 20],
      }
    );

    // Process descriptions
    if (batch.descriptions && batch.descriptions.length > 0) {
      // For English descriptions
      const enDesc = batch.descriptions.find((d) => d.language === "En");
      if (enDesc) {
        docDefinition.content.push({
          text: "English Analysis",
          style: "subheader",
        });

        if (enDesc.longDescription) {
          docDefinition.content.push(
            { text: "Briefed Analysis:", style: "contentHeader" },
            { text: enDesc.longDescription, margin: [0, 0, 0, 15] }
          );
        }

        if (enDesc.shortDescription) {
          docDefinition.content.push(
            { text: "Summarised Analysis:", style: "contentHeader" },
            { text: enDesc.shortDescription, margin: [0, 0, 0, 30] }
          );
        }
      }

      // For non-English descriptions
      const nonEnglishDescs = batch.descriptions.filter(
        (d) => d.language !== "En"
      );

      if (nonEnglishDescs.length > 0) {
        docDefinition.content.push({
          text: "Other Languages",
          style: "subheader",
          pageBreak: "before",
        });

        // Process each non-Latin description
        nonEnglishDescs.forEach((desc) => {
          const langName = languageDisplay[desc.language] || desc.language;

          docDefinition.content.push({
            text: `${langName} Analysis`,
            style: "contentHeader",
            margin: [0, 15, 0, 10],
          });

          if (desc.shortDescription) {
            docDefinition.content.push(
              { text: "Summarised Analysis:", style: "label" },
              {
                text: desc.shortDescription,
                margin: [0, 5, 0, 15],
                preserveLeadingSpaces: true,
              }
            );
          }

          if (desc.longDescription) {
            docDefinition.content.push(
              { text: "Briefed Analysis:", style: "label" },
              {
                text: desc.longDescription,
                margin: [0, 5, 0, 20],
                preserveLeadingSpaces: true,
              }
            );
          }
        });
      }
    } else if (batch.description) {
      // Legacy description support
      docDefinition.content.push(
        { text: "Analysis Description:", style: "subheader" },
        { text: batch.description }
      );
    }

    // Footer with web link
    docDefinition.content.push(
      { text: "Access Online", style: "subheader", pageBreak: "before" },
      { text: "For the best viewing experience, visit:", style: "note" },
      { text: batchUrl, link: batchUrl, style: "url", margin: [0, 5, 0, 20] },
      {
        style: "note",
        text: [
          "Note: ",
          "While this PDF attempts to display all languages, some complex scripts may not appear correctly depending on your PDF viewer. The web interface provides the most accurate display of all languages.",
        ],
      }
    );

    // Define styles
    docDefinition.styles = {
      header: {
        fontSize: 22,
        bold: true,
        marginBottom: 20,
      },
      subheader: {
        fontSize: 16,
        bold: true,
        marginBottom: 10,
        color: "#265C42",
      },
      contentHeader: {
        fontSize: 14,
        bold: true,
        marginTop: 10,
        marginBottom: 5,
      },
      label: {
        fontSize: 12,
        bold: true,
        marginTop: 8,
        marginBottom: 2,
      },
      url: {
        color: "blue",
        decoration: "underline",
      },
      note: {
        fontSize: 10,
        italics: true,
        color: "#444444",
      },
    };

    // Generate and download the PDF
    pdfMake
      .createPdf(docDefinition)
      .download(`${batch.name}_analysis_report.pdf`);

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
