const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient();

async function addTestDescriptions() {
  try {
    console.log("Starting script...");
    
    // Find a batch to add descriptions to
    const batch = await prisma.batch.findFirst({
      where: {
        id: "6976c48b773b626b38c3c38c" // From the URL in the screenshot
      },
      include: {
        descriptions: true,
        audioFiles: true
      }
    });

    if (!batch) {
      console.log("Batch not found");
      return;
    }

    console.log("Found batch:", batch.name);
    console.log("Existing descriptions:", batch.descriptions.length);
    console.log("Existing audio files:", batch.audioFiles.length);

    // Add test descriptions in multiple languages
    const testDescriptions = [
      {
        language: 'En',
        longDescription: 'Based on the analysis of drone images, the soybean crop shows significant concerns. Out of 7 images analyzed, 71.43% show signs of Soybean Mosaic disease, while only 28.57% appear healthy. The average confidence in our predictions is 90.8%, indicating high reliability in the analysis. The primary disease detected is Soybean Mosaic, which affects the majority of the crop area. Immediate intervention is recommended to prevent further spread of the disease.',
        shortDescription: 'Soybean crop analysis reveals 71.43% disease presence (primarily Soybean Mosaic) with intervention needed immediately.',
        wordCount: 95,
        confidence: 0.908
      },
      {
        language: 'Ta',
        longDescription: 'ட்ரோன் படங்களின் பகுப்பாய்வின் அடிப்படையில், சோயாபீன் பயிர் குறிப்பிடத்தக்க கவலைகளைக் காட்டுகிறது. பகுப்பாய்வு செய்யப்பட்ட 7 படங்களில், 71.43% சோயாபீன் மொசைக் நோயின் அறிகுறிகளைக் காட்டுகின்றன, அதே சமயம் 28.57% மட்டுமே ஆரோக்கியமாக தோன்றுகின்றன.',
        shortDescription: 'சோயாபீன் பயிர் பகுப்பாய்வில் 71.43% நோய் இருப்பு உடனடி தலையீடு தேவை.',
        wordCount: 67,
        confidence: 0.908
      },
      {
        language: 'Hi',
        longDescription: 'ड्रोन छवियों के विश्लेषण के आधार पर, सोयाबीन की फसल में महत्वपूर्ण चिंताएं दिख रही हैं। विश्लेषण की गई 7 छवियों में से, 71.43% में सोयाबीन मोज़ेक रोग के लक्षण दिख रहे हैं, जबकि केवल 28.57% स्वस्थ दिखाई दे रहे हैं।',
        shortDescription: 'सोयाबीन फसल विश्लेषण में 71.43% रोग की उपस्थिति, तत्काल हस्तक्षेप की आवश्यकता।',
        wordCount: 78,
        confidence: 0.908
      }
    ];

    // Create descriptions
    for (const desc of testDescriptions) {
      await prisma.description.create({
        data: {
          batchId: batch.id,
          ...desc
        }
      });
    }

    // Add test audio files
    const testAudioFiles = [
      {
        language: 'En',
        fileUrl: '/api/audio/sample-en.mp3',
        fileName: 'soybean_analysis_en.mp3',
        duration: 45.2,
        fileSize: 1024000
      },
      {
        language: 'Ta',
        fileUrl: '/api/audio/sample-ta.mp3',
        fileName: 'soybean_analysis_ta.mp3',
        duration: 52.1,
        fileSize: 1150000
      },
      {
        language: 'Hi',
        fileUrl: '/api/audio/sample-hi.mp3',
        fileName: 'soybean_analysis_hi.mp3',
        duration: 48.7,
        fileSize: 1080000
      }
    ];

    // Create audio files
    for (const audio of testAudioFiles) {
      await prisma.audioFile.create({
        data: {
          batchId: batch.id,
          ...audio
        }
      });
    }

    console.log("Test descriptions and audio files added successfully!");

  } catch (error) {
    console.error("Error adding test data:", error);
  } finally {
    await prisma.$disconnect();
  }
}

addTestDescriptions();
