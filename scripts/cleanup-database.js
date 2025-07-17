const { PrismaClient } = require("@prisma/client");
const prisma = new PrismaClient({
  log: ["query", "info", "warn", "error"],
});

async function cleanupDatabase() {
  console.log("Starting cleanup process...");
  try {
    console.log("Starting database cleanup...");

    // Step 1: Find orphaned descriptions (descriptions with non-existent batches)
    console.log("Finding orphaned descriptions...");
    const descriptions = await prisma.description.findMany();
    let orphanedDescriptions = [];

    for (const desc of descriptions) {
      const batch = await prisma.batch.findUnique({
        where: { id: desc.batchId },
      });

      if (!batch) {
        orphanedDescriptions.push(desc.id);
      }
    }

    if (orphanedDescriptions.length > 0) {
      console.log(
        `Found ${orphanedDescriptions.length} orphaned descriptions. Deleting...`
      );
      await prisma.description.deleteMany({
        where: {
          id: { in: orphanedDescriptions },
        },
      });
      console.log("Orphaned descriptions deleted.");
    } else {
      console.log("No orphaned descriptions found.");
    }

    // Step 2: Find orphaned audio files
    console.log("Finding orphaned audio files...");
    const audioFiles = await prisma.audioFile.findMany();
    let orphanedAudioFiles = [];

    for (const audio of audioFiles) {
      const batch = await prisma.batch.findUnique({
        where: { id: audio.batchId },
      });

      if (!batch) {
        orphanedAudioFiles.push(audio.id);
      }
    }

    if (orphanedAudioFiles.length > 0) {
      console.log(
        `Found ${orphanedAudioFiles.length} orphaned audio files. Deleting...`
      );
      await prisma.audioFile.deleteMany({
        where: {
          id: { in: orphanedAudioFiles },
        },
      });
      console.log("Orphaned audio files deleted.");
    } else {
      console.log("No orphaned audio files found.");
    }

    // Step 3: Add test data if needed
    const batch = await prisma.batch.findFirst({
      orderBy: { createdAt: "desc" },
      include: { descriptions: true, audioFiles: true },
    });

    if (batch) {
      console.log(`Found batch: ${batch.id} (${batch.name})`);
      console.log(`Descriptions: ${batch.descriptions.length}`);
      console.log(`Audio files: ${batch.audioFiles.length}`);

      // Add test description if none exist
      if (batch.descriptions.length === 0) {
        console.log("Adding test descriptions...");
        const languages = ["En", "Ta", "Hi"];

        for (const lang of languages) {
          await prisma.description.create({
            data: {
              batchId: batch.id,
              language: lang,
              longDescription: `This is a comprehensive analysis of ${batch.cropType} crop from drone imagery. The analysis shows healthy plant growth with good leaf density and proper spacing. The crop appears to be in good condition with minimal signs of stress or disease.`,
              shortDescription: `${batch.cropType} crop shows healthy growth with promising yield potential.`,
              wordCount: Math.floor(Math.random() * 100) + 50,
              confidence: Math.random() * 0.3 + 0.7, // Between 0.7 and 1.0
            },
          });
          console.log(`Created description for language: ${lang}`);
        }
      }

      // Add test audio files if none exist
      if (batch.audioFiles.length === 0) {
        console.log("Adding test audio files...");
        const languages = ["En", "Ta"];

        for (const lang of languages) {
          await prisma.audioFile.create({
            data: {
              batchId: batch.id,
              language: lang,
              fileUrl: `/sample-audio/${lang.toLowerCase()}_analysis.mp3`,
              fileName: `${lang.toLowerCase()}_analysis.mp3`,
              duration: 45.5,
              fileSize: 1024 * 1024, // 1MB
            },
          });
          console.log(`Created audio file for language: ${lang}`);
        }
      }

      // Update batch to mark descriptions and audio as completed
      await prisma.batch.update({
        where: { id: batch.id },
        data: {
          isDescCompleted: true,
          isAudioCompleted: true,
        },
      });
    } else {
      console.log("No batches found in the database.");
    }

    console.log("Database cleanup completed successfully!");
  } catch (error) {
    console.error("Error during database cleanup:", error);
  } finally {
    await prisma.$disconnect();
  }
}

cleanupDatabase();
