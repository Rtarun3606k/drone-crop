import { Router } from "express";
import { prisma } from "../config.mjs";

import { createJwtToken, refreshTokenIfNeeded } from "../lib/jwtTokens.mjs";
import { verifyToken } from "../middleware/midddleware.mjs";

const router = Router();

router.get("/get-user", verifyToken, refreshTokenIfNeeded, async (req, res) => {
  console.log("Fetching user details...");
  const userEmail = req.user.email;
  const user = await prisma.user.findUnique({
    where: {
      email: userEmail,
    },
  });

  if (!user) {
    return res.status(404).json({
      error: "User not found",
      message: "No user found with the provided email.",
    });
  }

  res.status(200).json({
    message: "User retrieved successfully",
    user: user,
  });
});

// Set or update home location
router.post("/set-home-location", verifyToken, async (req, res) => {
  try {
    const userId = req.user.userId || req.user.id; // support both
    if (!userId) {
      return res.status(401).json({ error: "Unauthorized. Please log in." });
    }

    const { coordinates, address, lat, lng } = req.body;

    // Handle both formats: new format {lat, lng, address} and old format {coordinates, address}
    let latitude, longitude, locationAddress;

    if (lat !== undefined && lng !== undefined) {
      latitude = lat;
      longitude = lng;
      locationAddress = address;
    } else if (coordinates && coordinates.latitude && coordinates.longitude) {
      latitude = coordinates.latitude;
      longitude = coordinates.longitude;
      locationAddress = address;
    } else {
      return res
        .status(400)
        .json({ error: "Coordinates and address are required." });
    }

    if (!latitude || !longitude || !locationAddress) {
      return res
        .status(400)
        .json({ error: "Coordinates and address are required." });
    }

    if (
      latitude < -90 ||
      latitude > 90 ||
      longitude < -180 ||
      longitude > 180
    ) {
      return res.status(400).json({
        error:
          "Invalid coordinate values. Latitude must be between -90 and 90, longitude between -180 and 180.",
      });
    }

    // Update user's home location in metadata
    const homeLocationData = {
      coordinates: {
        latitude: latitude,
        longitude: longitude,
        projected: (coordinates && coordinates.projected) || null,
      },
      address: locationAddress,
      setAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    // Fetch current user metadata
    const user = await prisma.user.findUnique({ where: { id: userId } });
    const currentMetadata = user?.metadata || {};

    const updatedUser = await prisma.user.update({
      where: { id: userId },
      data: {
        metadata: {
          ...currentMetadata,
          homeLocation: homeLocationData,
        },
      },
      select: {
        id: true,
        name: true,
        email: true,
        metadata: true,
      },
    });

    return res.status(200).json({
      success: true,
      message: "Home location set successfully!",
      homeLocation: homeLocationData,
    });
  } catch (error) {
    console.error("Error setting home location:", error);
    return res.status(500).json({
      error: "Failed to set home location. Please try again.",
    });
  }
});

// Get home location
router.get("/set-home-location", verifyToken, async (req, res) => {
  try {
    const userId = req.user.userId || req.user.id;
    if (!userId) {
      return res.status(401).json({ error: "Unauthorized. Please log in." });
    }

    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { metadata: true },
    });

    if (!user) {
      return res.status(404).json({ error: "User not found." });
    }

    const homeLocation = user.metadata?.homeLocation || null;

    // Transform the data format for frontend compatibility
    let coordinates = null;
    if (homeLocation && homeLocation.coordinates) {
      coordinates = {
        lat: homeLocation.coordinates.latitude,
        lng: homeLocation.coordinates.longitude,
        address: homeLocation.address,
      };
    }

    return res.status(200).json({
      success: true,
      coordinates: coordinates,
      homeLocation: homeLocation, // Keep the original structure for compatibility
    });
  } catch (error) {
    console.error("Error fetching home location:", error);
    return res.status(500).json({
      error: "Failed to fetch home location. Please try again.",
    });
  }
});

export default router;
