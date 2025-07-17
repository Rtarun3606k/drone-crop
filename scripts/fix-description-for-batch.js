const { PrismaClient } = require("@prisma/client");
const prisma = new PrismaClient();

// Get the ID from command line arguments
const batchId = process.argv[2];

if (!batchId) {
  console.error("Please provide a batch ID as argument");
  process.exit(1);
}

async function fixDescriptionsForBatch() {
  try {
    console.log(`Starting to fix descriptions for batch ${batchId}...`);

    // Step 1: Check if batch exists
    const batch = await prisma.batch.findUnique({
      where: { id: batchId },
      include: { descriptions: true },
    });

    if (!batch) {
      console.log(`Batch ${batchId} not found!`);
      process.exit(1);
    }

    console.log(`Found batch: ${batch.name} (${batch.preferredLanguage})`);
    console.log(`Current descriptions: ${batch.descriptions?.length || 0}`);

    // Step 2: Check for raw descriptions in the database
    const rawDescriptions = await prisma.$queryRaw`
      SELECT * FROM Description WHERE batchId = ${batchId}
    `;

    console.log(`Raw query found ${rawDescriptions?.length || 0} descriptions`);

    // Step 3: If we have descriptions data from MongoDB, add them to the system
    if (batch.descriptions.length === 0) {
      // Your MongoDB description data
      const descriptionData = [
        {
          batchId: batchId,
          language: "En",
          longDescription:
            "The overall crop health is assessed as poor, with only 25% of the soybean plants categorized as healthy. Two major threats, Soybean Semilooper pest attack and rust disease, have been identified. The Soybean Semilooper pest constitutes 50% of the class distribution and is the primary issue. Rust disease affects 25% of the crop, while no mosaic virus presence has been detected. Average confidence in predictions stands at 93.6%, signifying reliable results, though one analysis fell in the medium confidence range.\n\nImmediate action is essential to salvage the crop. For Soybean Semilooper control, integrated pest management methods should be implemented, including manual removal of larvae, pheromone traps, and timely spraying of approved insecticides such as Emamectin Benzoate. For rust control, fungicides like Triazoles are recommended combined with efforts to improve field drainage, which minimizes humid conditions favoring the disease. It is also vital to inspect the field promptly to verify predictions and apply treatments effectively.\n\nPreventive measures include crop rotation, use of disease-resistant soybean varieties, and constant monitoring for pests and diseases. Maintaining proper field hygiene and avoiding stress conditions like waterlogging are crucial to sustaining healthy crop growth. Farmers should conduct routine pest scouting and apply biological control agents like Trichogramma for integrated pest management.",
          shortDescription:
            "The soybean crop's health is poor due to a 50% semilooper pest attack and 25% rust presence. Immediate action involves pest management and fungicide application to control rust. Preventive measures include crop rotation, resistant varieties, and strict field monitoring.",
          wordCount: 246,
          confidence: 0.92,
        },
        {
          batchId: batchId,
          language: "Kn",
          longDescription:
            "ಒಟ್ಟಾರೆ ಬೆಳೆ ಆರೋಗ್ಯವನ್ನು ಕಳಪೆ ಎಂದು ನಿರ್ಣಯಿಸಲಾಗಿದೆ, ಕೇವಲ 25% ಸೋಯಾಬೀನ್ ಸಸ್ಯಗಳನ್ನು ಆರೋಗ್ಯಕರವೆಂದು ವರ್ಗೀಕರಿಸಲಾಗಿದೆ. ಸೋಯಾಬೀನ್ ಸೆಮಿಲೂಪರ್ ಕೀಟ ದಾಳಿ ಮತ್ತು ತುಕ್ಕು ರೋಗ ಎಂಬ ಎರಡು ಪ್ರಮುಖ ಬೆದರಿಕೆಗಳನ್ನು ಗುರುತಿಸಲಾಗಿದೆ. ಸೋಯಾಬೀನ್ ಸೆಮಿಲೂಪರ್ ಕೀಟವು ವರ್ಗ ವಿತರಣೆಯ 50% ರಷ್ಟಿದೆ ಮತ್ತು ಇದು ಪ್ರಾಥಮಿಕ ಸಮಸ್ಯೆಯಾಗಿದೆ. ತುಕ್ಕು ರೋಗವು 25% ಬೆಳೆಯ ಮೇಲೆ ಪರಿಣಾಮ ಬೀರುತ್ತದೆ, ಆದರೆ ಯಾವುದೇ ಮೊಸಾಯಿಕ್ ವೈರಸ್ ಉಪಸ್ಥಿತಿ ಕಂಡುಬಂದಿಲ್ಲ. ಭವಿಷ್ಯವಾಣಿಗಳಲ್ಲಿ ಸರಾಸರಿ ವಿಶ್ವಾಸವು 93.6% ರಷ್ಟಿದೆ, ಇದು ವಿಶ್ವಾಸಾರ್ಹ ಫಲಿತಾಂಶಗಳನ್ನು ಸೂಚಿಸುತ್ತದೆ, ಆದರೂ ಒಂದು ವಿಶ್ಲೇಷಣೆಯು ಮಧ್ಯಮ ವಿಶ್ವಾಸದ ವ್ಯಾಪ್ತಿಯಲ್ಲಿ ಕುಸಿದಿದೆ.\n\nಬೆಳೆಯನ್ನು ಉಳಿಸಲು ತಕ್ಷಣದ ಕ್ರಮ ಅತ್ಯಗತ್ಯ. ಸೋಯಾಬೀನ್ ಸೆಮಿಲೂಪರ್ ನಿಯಂತ್ರಣಕ್ಕಾಗಿ, ಲಾರ್ವಾಗಳನ್ನು ಹಸ್ತಚಾಲಿತವಾಗಿ ತೆಗೆದುಹಾಕುವುದು, ಫೆರೊಮೋನ್ ಬಲೆಗಳು ಮತ್ತು ಎಮಾಮೆಕ್ಟಿನ್ ಬೆಂಜೊಯೇಟ್ ನಂತಹ ಅನುಮೋದಿತ ಕೀಟನಾಶಕಗಳನ್ನು ಸಮಯೋಚಿತವಾಗಿ ಸಿಂಪಡಿಸುವುದು ಸೇರಿದಂತೆ ಸಮಗ್ರ ಕೀಟ ನಿರ್ವಹಣಾ ವಿಧಾನಗಳನ್ನು ಜಾರಿಗೆ ತರಬೇಕು. ತುಕ್ಕು ನಿಯಂತ್ರಣಕ್ಕಾಗಿ, ಟ್ರಯಾಜೋಲ್ ಗಳಂತಹ ಶಿಲೀಂಧ್ರನಾಶಕಗಳನ್ನು ಕ್ಷೇತ್ರ ಒಳಚರಂಡಿಯನ್ನು ಸುಧಾರಿಸುವ ಪ್ರಯತ್ನಗಳೊಂದಿಗೆ ಸಂಯೋಜಿಸಲು ಶಿಫಾರಸು ಮಾಡಲಾಗುತ್ತದೆ, ಇದು ರೋಗಕ್ಕೆ ಅನುಕೂಲಕರವಾದ ಆರ್ದ್ರ ಪರಿಸ್ಥಿತಿಗಳನ್ನು ಕಡಿಮೆ ಮಾಡುತ್ತದೆ. ಊಹೆಗಳನ್ನು ಪರಿಶೀಲಿಸಲು ಮತ್ತು ಚಿಕಿತ್ಸೆಗಳನ್ನು ಪರಿಣಾಮಕಾರಿಯಾಗಿ ಅನ್ವಯಿಸಲು ಕ್ಷೇತ್ರವನ್ನು ತ್ವರಿತವಾಗಿ ಪರಿಶೀಲಿಸುವುದು ಸಹ ಅತ್ಯಗತ್ಯ.\n\nತಡೆಗಟ್ಟುವ ಕ್ರಮಗಳಲ್ಲಿ ಬೆಳೆ ತಿರುಗುವಿಕೆ, ರೋಗ-ನಿರೋಧಕ ಸೋಯಾಬೀನ್ ಪ್ರಭೇದಗಳ ಬಳಕೆ ಮತ್ತು ಕೀಟಗಳು ಮತ್ತು ರೋಗಗಳ ನಿರಂತರ ಮೇಲ್ವಿಚಾರಣೆ ಸೇರಿವೆ. ಸರಿಯಾದ ಹೊಲದ ನೈರ್ಮಲ್ಯವನ್ನು ಕಾಪಾಡಿಕೊಳ್ಳುವುದು ಮತ್ತು ಜಲಾವೃತತೆಯಂತಹ ಒತ್ತಡದ ಪರಿಸ್ಥಿತಿಗಳನ್ನು ತಪ್ಪಿಸುವುದು ಆರೋಗ್ಯಕರ ಬೆಳೆ ಬೆಳವಣಿಗೆಯನ್ನು ಉಳಿಸಿಕೊಳ್ಳಲು ನಿರ್ಣಾಯಕವಾಗಿದೆ. ರೈತರು ವಾಡಿಕೆಯ ಕೀಟ ಸ್ಕೌಟಿಂಗ್ ನಡೆಸಬೇಕು ಮತ್ತು ಸಮಗ್ರ ಕೀಟ ನಿರ್ವಹಣೆಗಾಗಿ ಟ್ರೈಕೋಗ್ರಾಮ್ಮಾದಂತಹ ಜೈವಿಕ ನಿಯಂತ್ರಣ ಏಜೆಂಟ್ ಗಳನ್ನು ಅನ್ವಯಿಸಬೇಕು.",
          shortDescription:
            "50% ಸೆಮಿಲೂಪರ್ ಕೀಟ ದಾಳಿ ಮತ್ತು 25% ತುಕ್ಕು ಉಪಸ್ಥಿತಿಯಿಂದಾಗಿ ಸೋಯಾಬೀನ್ ಬೆಳೆಯ ಆರೋಗ್ಯವು ಕಳಪೆಯಾಗಿದೆ. ತಕ್ಷಣದ ಕ್ರಮವು ತುಕ್ಕು ನಿಯಂತ್ರಿಸಲು ಕೀಟ ನಿರ್ವಹಣೆ ಮತ್ತು ಶಿಲೀಂಧ್ರನಾಶಕದ ಬಳಕೆಯನ್ನು ಒಳಗೊಂಡಿರುತ್ತದೆ. ತಡೆಗಟ್ಟುವ ಕ್ರಮಗಳಲ್ಲಿ ಬೆಳೆ ತಿರುಗುವಿಕೆ, ನಿರೋಧಕ ಪ್ರಭೇದಗಳು ಮತ್ತು ಕಟ್ಟುನಿಟ್ಟಾದ ಕ್ಷೇತ್ರ ಮೇಲ್ವಿಚಾರಣೆ ಸೇರಿವೆ.",
          wordCount: 197,
          confidence: 0.89,
        },
      ];

      // Create descriptions
      console.log("Adding descriptions to the database...");

      for (const desc of descriptionData) {
        const created = await prisma.description.create({
          data: desc,
        });
        console.log(`Created description in ${desc.language}: ${created.id}`);
      }

      console.log("Descriptions added successfully!");
    }

    // Step 4: Update batch to mark description as completed
    await prisma.batch.update({
      where: { id: batchId },
      data: {
        isDescCompleted: true,
      },
    });

    console.log("Batch updated successfully!");
  } catch (error) {
    console.error("Error:", error);
  } finally {
    await prisma.$disconnect();
  }
}

fixDescriptionsForBatch();
