/**
 * PDF Generator utility for multilingual content
 * Handles non-Latin scripts like Kannada, Tamil, etc. using HTML canvas rendering
 */

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
 * @param {Object} batch - Batch object with descriptions
 * @param {Object} languageDisplay - Map of language codes to display names
 * @param {string} locale - Current locale for date formatting
 */
export async function generateAnalysisPDF(
  batch,
  languageDisplay,
  locale = "en"
) {
  try {
    // Import required libraries
    const { jsPDF } = await import("jspdf");
    const html2canvas = (await import("html2canvas")).default;

    // Create a new PDF document
    const doc = new jsPDF();

    // Set initial position
    let yPosition = 20;
    const pageHeight = doc.internal.pageSize.height;
    const margin = 20;
    const lineHeight = 10;

    // Title
    doc.setFontSize(20);
    doc.setFont("helvetica", "bold");
    doc.text(`Crop Analysis Report: ${batch.name}`, margin, yPosition);
    yPosition += 20;

    // Batch Info
    doc.setFontSize(12);
    doc.setFont("helvetica", "normal");
    doc.text(`Crop Type: ${batch.cropType}`, margin, yPosition);
    yPosition += lineHeight;
    doc.text(`Images Count: ${batch.imagesCount}`, margin, yPosition);
    yPosition += lineHeight;
    doc.text(
      `Created: ${formatPdfDate(batch.createdAt, locale)}`,
      margin,
      yPosition
    );
    yPosition += lineHeight * 2;

    // Process descriptions
    if (batch.descriptions && batch.descriptions.length > 0) {
      // Create a hidden container for rendering multilingual content
      const container = document.createElement("div");
      container.style.position = "absolute";
      container.style.left = "-9999px";
      container.style.top = "-9999px";
      container.style.width = "500px";
      container.style.fontFamily = "Arial, sans-serif";
      container.style.backgroundColor = "white";
      container.style.color = "black";
      document.body.appendChild(container);

      // Process each description
      for (const desc of batch.descriptions) {
        // Add a new page if not enough space
        if (yPosition > pageHeight - 60) {
          doc.addPage();
          yPosition = margin;
        }

        // Language title (rendered directly with jsPDF)
        doc.setFontSize(16);
        doc.setFont("helvetica", "bold");
        doc.text(
          `${languageDisplay[desc.language] || desc.language}`,
          margin,
          yPosition
        );
        yPosition += lineHeight * 1.5;

        // Title for long description
        doc.setFontSize(14);
        doc.setFont("helvetica", "bold");
        doc.text("Briefed Analysis:", margin, yPosition);
        yPosition += lineHeight * 1.5;

        // For non-Latin scripts (like Kannada, Tamil, etc.), use HTML rendering
        if (desc.language !== "En") {
          // Clear previous content
          container.innerHTML = "";

          // Set content for rendering
          container.innerHTML = `
            <div style="font-size: 14px; padding: 10px; line-height: 1.5;">
              ${desc.longDescription || "No description available"}
            </div>
          `;

          // Convert the HTML element to an image using html2canvas
          const canvas = await html2canvas(container, {
            scale: 2,
            useCORS: true,
            logging: false,
          });

          // Convert canvas to image data
          const imgData = canvas.toDataURL("image/png");

          // Calculate appropriate width to fit on page
          const imgWidth = 170;
          const imgHeight = (canvas.height * imgWidth) / canvas.width;

          // Add image to PDF
          doc.addImage(imgData, "PNG", margin, yPosition, imgWidth, imgHeight);

          // Update position
          yPosition += imgHeight + lineHeight;
        } else {
          // For English, use direct text rendering
          doc.setFontSize(10);
          doc.setFont("helvetica", "normal");

          const longDesc = desc.longDescription || "No description available";
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

        // Short description section
        if (desc.shortDescription) {
          // Check if we need a new page
          if (yPosition > pageHeight - 40) {
            doc.addPage();
            yPosition = margin;
          }

          // Title for short description
          doc.setFontSize(14);
          doc.setFont("helvetica", "bold");
          doc.text("Summarised Analysis:", margin, yPosition);
          yPosition += lineHeight * 1.5;

          // For non-Latin scripts, use HTML rendering again
          if (desc.language !== "En") {
            // Clear previous content
            container.innerHTML = "";

            // Set content for rendering
            container.innerHTML = `
              <div style="font-size: 14px; padding: 10px; line-height: 1.5;">
                ${desc.shortDescription}
              </div>
            `;

            // Convert the HTML element to an image
            const canvas = await html2canvas(container, {
              scale: 2,
              useCORS: true,
              logging: false,
            });

            // Convert canvas to image data
            const imgData = canvas.toDataURL("image/png");

            // Calculate appropriate width
            const imgWidth = 170;
            const imgHeight = (canvas.height * imgWidth) / canvas.width;

            // Add image to PDF
            doc.addImage(
              imgData,
              "PNG",
              margin,
              yPosition,
              imgWidth,
              imgHeight
            );

            // Update position
            yPosition += imgHeight + lineHeight;
          } else {
            // For English, use direct text rendering
            doc.setFontSize(10);
            doc.setFont("helvetica", "normal");

            const summaryLines = doc.splitTextToSize(
              desc.shortDescription,
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
        }

        // Add space between descriptions
        yPosition += lineHeight * 2;
      }

      // Clean up the temporary container
      document.body.removeChild(container);
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

    // Save the PDF
    doc.save(`${batch.name}_analysis_report.pdf`);

    return true;
  } catch (error) {
    console.error("Error generating PDF:", error);
    return false;
  }
}
