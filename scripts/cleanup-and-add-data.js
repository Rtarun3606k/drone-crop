const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient();

async function cleanupAndAddTestData() {
  try {
    console.log('Starting cleanup and test data creation...');
    
    // Step 1: Clean up orphaned records
    console.log('1. Cleaning up orphaned Description records...');
    const orphanedDescriptions = await prisma.description.findMany({
      where: {
        batch: null
      }
    });
    
    if (orphanedDescriptions.length > 0) {
      console.log(`Found ${orphanedDescriptions.length} orphaned descriptions, deleting...`);
      await prisma.description.deleteMany({
        where: {
          batch: null
        }
      });
    }
    
    console.log('2. Cleaning up orphaned AudioFile records...');
    const orphanedAudio = await prisma.audioFile.findMany({
      where: {
        batch: null
      }
    });
    
    if (orphanedAudio.length > 0) {
      console.log(`Found ${orphanedAudio.length} orphaned audio files, deleting...`);
      await prisma.audioFile.deleteMany({
        where: {
          batch: null
        }
      });
    }
    
    // Step 2: Find an existing batch
    console.log('3. Finding an existing batch...');
    const batch = await prisma.batch.findFirst({
      orderBy: { createdAt: 'desc' },
      include: {
        descriptions: true,
        audioFiles: true
      }
    });
    
    if (!batch) {
      console.log('No batches found in database');
      return;
    }
    
    console.log(`Found batch: ${batch.id} - ${batch.name}`);
    console.log(`Existing descriptions: ${batch.descriptions.length}`);
    console.log(`Existing audio files: ${batch.audioFiles.length}`);
    
    // Step 3: Add test descriptions if none exist
    if (batch.descriptions.length === 0) {
      console.log('4. Adding test descriptions...');
      
      const descriptions = [
        {
          batchId: batch.id,
          language: 'En',
          longDescription: `This comprehensive analysis of ${batch.cropType} crop from drone imagery reveals healthy plant growth patterns with optimal leaf density and proper spatial distribution. The vegetation shows strong chlorophyll content indicating good photosynthetic activity. Plant spacing appears uniform across the field, suggesting effective planting practices. No significant signs of disease, pest damage, or nutrient deficiency are visible in the analyzed imagery. The crop canopy coverage is excellent, indicating good establishment and growth vigor. Overall, the ${batch.cropType} crop demonstrates promising yield potential based on current health indicators and growth characteristics observed in the drone survey.`,
          shortDescription: `${batch.cropType} crop displays excellent health with uniform growth, strong leaf density, and promising yield potential. No disease or stress indicators detected.`,
          wordCount: 95,
          confidence: 0.92
        },
        {
          batchId: batch.id,
          language: 'Ta',
          longDescription: `இந்த ட்ரோன் படங்களில் இருந்து ${batch.cropType} பயிரின் விரிவான பகுப்பாய்வு ஆரோக்கியமான தாவர வளர்ச்சி முறைகளை வெளிப்படுத்துகிறது. இலைகளின் அடர்த்தி சிறந்தது மற்றும் தாவரங்களின் இடைவெளி சரியானது. பயிர் நல்ல நிலையில் உள்ளது மற்றும் நோய் அல்லது பூச்சித் தாக்குதலின் எந்த அறிகுறியும் இல்லை. ஒட்டுமொத்தமாக, ${batch.cropType} பயிர் நல்ல மகசூல் திறனைக் காட்டுகிறது.`,
          shortDescription: `${batch.cropType} பயிர் சிறந்த ஆரோக்கியத்துடன் சீரான வளர்ச்சி மற்றும் நல்ல மகசூல் வாய்ப்புகளைக் காட்டுகிறது.`,
          wordCount: 85,
          confidence: 0.88
        },
        {
          batchId: batch.id,
          language: 'Hi',
          longDescription: `ड्रोन इमेजरी से ${batch.cropType} फसल का यह व्यापक विश्लेषण स्वस्थ पौधे की वृद्धि पैटर्न को दर्शाता है। पत्तियों का घनत्व उत्कृष्ट है और पौधों की दूरी उचित है। फसल अच्छी स्थिति में है और कोई बीमारी या कीट क्षति के संकेत नहीं हैं। समग्र रूप से, ${batch.cropType} फसल अच्छी उपज क्षमता दिखाती है।`,
          shortDescription: `${batch.cropType} फसल उत्कृष्ट स्वास्थ्य के साथ समान वृद्धि और अच्छी उपज संभावना दिखाती है।`,
          wordCount: 70,
          confidence: 0.90
        }
      ];
      
      for (const desc of descriptions) {
        const created = await prisma.description.create({
          data: desc
        });
        console.log(`   ✓ Created ${desc.language} description: ${created.id}`);
      }
    }
    
    // Step 4: Add test audio files if none exist
    if (batch.audioFiles.length === 0) {
      console.log('5. Adding test audio files...');
      
      const audioFiles = [
        {
          batchId: batch.id,
          language: 'En',
          fileUrl: '/audio/sample_en_analysis.mp3',
          fileName: 'english_analysis.mp3',
          duration: 45.5,
          fileSize: 1024000
        },
        {
          batchId: batch.id,
          language: 'Ta',
          fileUrl: '/audio/sample_ta_analysis.mp3',
          fileName: 'tamil_analysis.mp3',
          duration: 42.3,
          fileSize: 980000
        },
        {
          batchId: batch.id,
          language: 'Hi',
          fileUrl: '/audio/sample_hi_analysis.mp3',
          fileName: 'hindi_analysis.mp3',
          duration: 40.8,
          fileSize: 950000
        }
      ];
      
      for (const audio of audioFiles) {
        const created = await prisma.audioFile.create({
          data: audio
        });
        console.log(`   ✓ Created ${audio.language} audio file: ${created.id}`);
      }
    }
    
    console.log('\n✅ Database cleanup and test data creation completed successfully!');
    console.log(`\nYou can now visit the batch detail page to see the descriptions and audio files:`);
    console.log(`Batch ID: ${batch.id}`);
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await prisma.$disconnect();
  }
}

cleanupAndAddTestData();
