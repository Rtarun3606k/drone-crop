console.log('Testing Prisma connection...');

const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

async function test() {
  try {
    // Test basic connection
    const count = await prisma.batch.count();
    console.log(`Found ${count} batches in database`);
    
    // Get latest batch
    const batch = await prisma.batch.findFirst({
      orderBy: { createdAt: 'desc' }
    });
    
    if (batch) {
      console.log(`Latest batch: ${batch.id} - ${batch.name}`);
      
      // Add a simple description
      const desc = await prisma.description.create({
        data: {
          batchId: batch.id,
          language: 'En',
          longDescription: 'This is a test description for the crop analysis.',
          shortDescription: 'Test summary of crop analysis.',
          wordCount: 10,
          confidence: 0.9
        }
      });
      console.log(`Created description: ${desc.id}`);
    }
    
  } catch (error) {
    console.error('Error:', error.message);
  } finally {
    await prisma.$disconnect();
  }
}

test();
