import { Router } from "express";
import { prisma } from "../config.mjs";
import { createJwtToken } from "../lib/jwtTokens.mjs";

const router = Router();

// Health check endpoint
router.get("/health", (req, res) => {
  res.status(200).json({
    status: "OK",
    message: "Mobile Backend API is running",
    timestamp: new Date().toISOString(),
  });
});

// API info endpoint
router.get("/", (req, res) => {
  res.json({
    auth: "Authentication API",
  });
});

router.post("/login", async (req, res) => {
  const { email, mobileId } = req.body;
  if (!email || !mobileId) {
    return res.status(400).json({
      error: "Missing required fields",
      message: "Email and mobileId are required for login.",
    });
  }
  try {
    const user = await prisma.user.findUnique({
      where: {
        email: email,
        mobileId: mobileId,
      },
    });

    if (!user) {
      return res.status(404).json({
        error: "User not found",
        message: "No user found with the provided email and mobileId.",
      });
    }

    // Here you would typically generate a token or session for the user
    const token = createJwtToken(user);
    res.status(200).json({
      message: "Login successful",
      accessToken: token,
    });
  } catch (error) {
    console.error("Login error:", error);
    res.status(500).json({
      error: "Internal Server Error",
      message: "An error occurred while processing your request.",
    });
  }
});

export default router;
