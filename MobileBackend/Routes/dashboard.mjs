import { Router } from "express";
import multer from "multer";
import { storeZip } from "../lib/storage.mjs";
import { verifyToken } from "../middleware/midddleware.mjs";
import { prisma } from "../config.mjs";
import path from "path";
import fs from "fs";
import { refreshTokenIfNeeded } from "../lib/jwtTokens.mjs";

const router = Router();

// Configure multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 100 * 1024 * 1024, // 100MB limit
  },
  fileFilter: (req, file, cb) => {
    if (
      file.mimetype === "application/zip" ||
      file.originalname.endsWith(".zip")
    ) {
      cb(null, true);
    } else {
      cb(new Error("Only ZIP files are allowed"), false);
    }
  },
});

const mapLocaleToLanguageEnum = (locale) => {
  const mapping = {
    en: "En",
    ta: "Ta",
    hi: "Hi",
    te: "Te",
    ml: "Ml",
    kn: "Kn",
  };
  return mapping[locale?.toLowerCase()] || "En";
};

// POST route for batch upload
router.post(
  "/upload-batch",
  verifyToken,
  upload.single("imagesZip"),
  async (req, res) => {
    try {
      const {
        batchName,
        cropType,
        imagesCount,
        preferredLanguage: clientLanguage,
        metadata,
      } = req.body;
      const zipFile = req.file;

      // Use userId from token (not id)
      const userId = req.user && req.user.userId;
      if (!userId) {
        return res.status(401).json({
          error: "Unauthorized",
          message: "User ID missing in token.",
        });
      }

      // Validate required fields
      if (!zipFile) {
        return res.status(400).json({
          error: "ZIP file missing or invalid.",
          message: "Please upload a valid ZIP file.",
        });
      }

      if (!batchName || !cropType) {
        return res.status(400).json({
          error: "Missing required fields",
          message: "batchName and cropType are required.",
        });
      }

      // Parse metadata if provided
      let parsedMetadata = null;
      if (metadata) {
        try {
          parsedMetadata = JSON.parse(metadata);
        } catch (error) {
          return res.status(400).json({
            error: "Invalid metadata format",
            message: "Metadata must be valid JSON.",
          });
        }
      }

      // Store the ZIP file
      const uploadResult = await storeZip(zipFile, zipFile.originalname);

      if (!uploadResult.success) {
        return res.status(500).json({
          error: "Failed to store ZIP file.",
          message: uploadResult.error || "Unknown storage error",
        });
      }

      // Map language preference
      const preferredLanguage = mapLocaleToLanguageEnum(clientLanguage);

      // Create batch in database
      const batch = await prisma.batch.create({
        data: {
          name: batchName,
          cropType: cropType,
          imagesZipURL: uploadResult.url,
          imagesCount: parseInt(imagesCount) || 0,
          userId: userId, // Use userId from token
          preferredLanguage: preferredLanguage,
          metadata: parsedMetadata,
        },
      });

      if (!batch) {
        return res.status(500).json({
          error: "Failed to save batch data.",
          message: "Database operation failed.",
        });
      }

      console.log("✅ Batch data saved to database:", {
        id: batch.id,
        name: batch.name,
        cropType: batch.cropType,
        fileSize: `${(zipFile.size / (1024 * 1024)).toFixed(2)} MB`,
        imagesCount: batch.imagesCount,
        preferredLanguage: batch.preferredLanguage,
        userId: batch.userId,
        sessionId: batch.sessionId,
      });

      res.status(200).json({
        message: "Upload received and processed successfully!",
        batch: {
          id: batch.id,
          name: batch.name,
          cropType: batch.cropType,
          imagesZipURL: batch.imagesZipURL,
          imagesCount: batch.imagesCount,
          userId: batch.userId,
          preferredLanguage: batch.preferredLanguage,
          metadata: batch.metadata,
          sessionId: batch.sessionId,
          createdAt: batch.createdAt,
          isModelCompleted: batch.isModelCompleted,
          isDescCompleted: batch.isDescCompleted,
          isAudioCompleted: batch.isAudioCompleted,
          hasExecutionFailed: batch.hasExecutionFailed,
        },
        uploadInfo: {
          originalName: zipFile.originalname,
          size: zipFile.size,
          storedPath: uploadResult.path,
        },
      });
    } catch (error) {
      console.error("Batch upload error:", error);

      if (error.code === "LIMIT_FILE_SIZE") {
        return res.status(400).json({
          error: "File too large",
          message: "ZIP file must be less than 100MB.",
        });
      }

      res.status(500).json({
        error: "Internal Server Error",
        message: "An error occurred while processing your batch upload.",
      });
    }
  }
);

// GET route to fetch user batches
router.get("/batches", verifyToken, async (req, res) => {
  try {
    const userId = req.user.id;

    // Fetch all batches for the current user (simplified like Next.js version)
    const batches = await prisma.batch.findMany({
      where: {
        userId: userId,
      },
      orderBy: {
        createdAt: "desc",
      },
    });

    return res.status(200).json({
      batches,
    });
  } catch (error) {
    console.error("Error fetching batches:", error);
    res.status(500).json({
      error: "Failed to fetch batches.",
      message: "An error occurred while fetching batches.",
    });
  }
});

// GET route to fetch specific batch details
router.get("/batch/:id", verifyToken, async (req, res) => {
  try {
    const { id } = req.params;
    const userId = req.user.id;

    // Validate the ID format for MongoDB ObjectId
    if (!id.match(/^[0-9a-fA-F]{24}$/)) {
      return res.status(400).json({
        error: "Invalid batch ID format.",
      });
    }

    // Fetch the specific batch with related audioFiles and descriptions
    const batch = await prisma.batch.findUnique({
      where: {
        id: id,
        userId: userId,
      },
      include: {
        audioFiles: true,
        descriptions: true, // ✅ Fetch descriptions directly here
      },
    });

    if (!batch) {
      return res.status(404).json({
        error: "Batch not found or you don't have permission to view it.",
      });
    }
    // console.log("✅ Fetched batch details:", batch);

    return res.status(200).json({
      batch,
    });
  } catch (error) {
    console.error("Error fetching batch details:", error);
    res.status(500).json({
      error: "Failed to fetch batch details.",
      message: "An error occurred while fetching the batch.",
    });
  }
});

// In your dashboard routes file (e.g., /routes/dashboard.js)
// Make sure to import 'path' and 'fs' at the top of the file

// ... your other routes

// --- Secure Audio Streaming Route ---
// This route streams an audio file associated with a specific batch ID.
router.get("/audio/:batchId/:userId", async (req, res) => {
  console.log("🔒 Verifying token and streaming audio...");

  try {
    const { batchId } = req.params;
    const { userId } = req.params; // Assuming verifyToken adds user object to req
    console.log("🔍 Streaming audio for batch ID:", batchId);

    // 1. Find the batch in the database
    const batch = await prisma.batch.findUnique({
      where: { id: batchId },
    });

    // 2. Security Check: Ensure batch exists and belongs to the logged-in user
    if (!batch) {
      return res.status(404).json({ error: "Batch not found." });
    }
    if (batch.userId !== userId) {
      return res
        .status(403)
        .json({ error: "Forbidden: You do not have access to this batch." });
    }

    // 3. Check if there is an audioURL to serve
    if (!batch.audioURL) {
      return res
        .status(404)
        .json({ error: "No audio file associated with this batch." });
    }

    // 4. Construct the absolute file path
    // IMPORTANT: This path is constructed relative to your project structure.
    // It goes up one level from MobileBackend to the drone-crop root, then into public.
    const audioFilePath =
      "/home/dragoon/coding/drone-crop/public" + batch.audioURL; // Go up from /MobileBackend/routes to /drone-crop, then into /public
    // Append the path from DB, e.g., /audioFiles/xyz.mp3

    // A more robust way if your script's location might change:
    // const projectRoot = path.resolve(__dirname, '../../../');
    // const audioFilePath = path.join(projectRoot, 'public', batch.audioURL);

    // 5. Check if the file exists on the server's disk
    if (!fs.existsSync(audioFilePath)) {
      console.error(`File not found at path: ${audioFilePath}`);
      return res.status(404).json({ error: "Audio file not found on server." });
    }

    // 6. Stream the file to the client
    res.setHeader("Content-Type", "audio/mpeg");
    const stream = fs.createReadStream(audioFilePath);
    stream.pipe(res);
  } catch (error) {
    console.error("Error streaming audio file:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// ... rest of your router exports

export default router;
