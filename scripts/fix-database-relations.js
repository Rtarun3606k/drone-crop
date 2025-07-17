// Database relation fix utility
// Run with: node scripts/fix-database-relations.js

import { PrismaClient } from "@prisma/client";
const prisma = new PrismaClient();

async function main() {
  console.log("Starting database relation fix utility...");

  // 1. Find all batches
  console.log("Fetching all batches...");
  const batches = await prisma.batch.findMany();
  console.log(`Found ${batches.length} batches`);

  // 2. For each batch, ensure descriptions have proper relations
  let fixedDescriptions = 0;
  let orphanedDescriptions = 0;

  for (const batch of batches) {
    // Find descriptions that should belong to this batch
    const descriptions = await prisma.description.findMany({
      where: {
        batchId: batch.id,
      },
    });

    console.log(`Batch ${batch.id}: Found ${descriptions.length} descriptions`);

    // If the batch has no descriptions but has a legacy description field, create a new description
    if (descriptions.length === 0 && batch.description) {
      console.log(
        `Batch ${batch.id}: Creating new description from legacy field`
      );
      try {
        await prisma.description.create({
          data: {
            batchId: batch.id,
            language: batch.preferredLanguage,
            longDescription: batch.description,
            shortDescription: "", // Empty short description since it doesn't exist in legacy format
            confidence: 1.0,
            wordCount: batch.description.split(" ").length,
          },
        });
        fixedDescriptions++;
      } catch (error) {
        console.error(
          `Error creating description for batch ${batch.id}:`,
          error
        );
      }
    }
  }

  // 3. Find orphaned descriptions (those without a valid batch)
  console.log("Checking for orphaned descriptions...");
  const allDescriptions = await prisma.description.findMany();

  for (const desc of allDescriptions) {
    const batch = await prisma.batch.findUnique({
      where: { id: desc.batchId },
    });

    if (!batch) {
      console.log(
        `Orphaned description found: ${desc.id} with batchId ${desc.batchId}`
      );
      orphanedDescriptions++;

      // Uncomment to delete orphaned descriptions
      // await prisma.description.delete({
      //   where: { id: desc.id }
      // });
    }
  }

  // 4. Summary
  console.log("\nDatabase Repair Summary:");
  console.log(`- Total batches: ${batches.length}`);
  console.log(`- Fixed descriptions: ${fixedDescriptions}`);
  console.log(`- Orphaned descriptions: ${orphanedDescriptions}`);

  if (orphanedDescriptions > 0) {
    console.log(
      "\nNote: To delete orphaned descriptions, uncomment the deletion code in this script."
    );
  }

  console.log("\nFinished database relation fix utility");
}

main()
  .catch((e) => {
    console.error("Error in database repair script:", e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
