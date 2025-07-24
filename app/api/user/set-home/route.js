import { auth } from "@/app/auth";
import { prisma } from "@/app/lib/prisma-server";
import { NextResponse } from "next/server";

export async function POST(request) {
  try {
    const session = await auth();

    if (!session || !session.user) {
      return NextResponse.json(
        { error: "Unauthorized. Please log in." },
        { status: 401 }
      );
    }

    const data = await request.json();
    const { coordinates, address, lat, lng } = data;

    // Handle both formats: new format {lat, lng, address} and old format {coordinates, address}
    let latitude, longitude, locationAddress;

    if (lat !== undefined && lng !== undefined) {
      // New format: {lat, lng, address}
      latitude = lat;
      longitude = lng;
      locationAddress = address;
    } else if (coordinates && coordinates.latitude && coordinates.longitude) {
      // Old format: {coordinates: {latitude, longitude}, address}
      latitude = coordinates.latitude;
      longitude = coordinates.longitude;
      locationAddress = address;
    } else {
      return NextResponse.json(
        { error: "Coordinates and address are required." },
        { status: 400 }
      );
    }

    // Validate required fields
    if (!latitude || !longitude || !locationAddress) {
      return NextResponse.json(
        { error: "Coordinates and address are required." },
        { status: 400 }
      );
    }

    // Validate coordinate ranges
    if (
      latitude < -90 ||
      latitude > 90 ||
      longitude < -180 ||
      longitude > 180
    ) {
      return NextResponse.json(
        {
          error:
            "Invalid coordinate values. Latitude must be between -90 and 90, longitude between -180 and 180.",
        },
        { status: 400 }
      );
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

    const updatedUser = await prisma.user.update({
      where: { id: session.user.id },
      data: {
        metadata: {
          ...session.user.metadata,
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

    return NextResponse.json(
      {
        success: true,
        message: "Home location set successfully!",
        homeLocation: homeLocationData,
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Error setting home location:", error);
    return NextResponse.json(
      { error: "Failed to set home location. Please try again." },
      { status: 500 }
    );
  }
}

export async function GET(request) {
  try {
    const session = await auth();

    if (!session || !session.user) {
      return NextResponse.json(
        { error: "Unauthorized. Please log in." },
        { status: 401 }
      );
    }

    const user = await prisma.user.findUnique({
      where: { id: session.user.id },
      select: {
        metadata: true,
      },
    });

    if (!user) {
      return NextResponse.json({ error: "User not found." }, { status: 404 });
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

    return NextResponse.json(
      {
        success: true,
        coordinates: coordinates,
        homeLocation: homeLocation, // Keep original format for backward compatibility
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Error fetching home location:", error);
    return NextResponse.json(
      { error: "Failed to fetch home location." },
      { status: 500 }
    );
  }
}

/**
 * @swagger
 * /user/set-home:
 *   post:
 *     summary: Set user's home location
 *     description: Allows an authenticated user to set their home location using coordinates and an address. Supports both old and new formats.
 *     tags:
 *       - User
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             oneOf:
 *               - type: object
 *                 required: [lat, lng, address]
 *                 properties:
 *                   lat:
 *                     type: number
 *                     format: float
 *                   lng:
 *                     type: number
 *                     format: float
 *                   address:
 *                     type: string
 *               - type: object
 *                 required: [coordinates, address]
 *                 properties:
 *                   coordinates:
 *                     type: object
 *                     required: [latitude, longitude]
 *                     properties:
 *                       latitude:
 *                         type: number
 *                       longitude:
 *                         type: number
 *                       projected:
 *                         type: object
 *                         nullable: true
 *                   address:
 *                     type: string
 *     responses:
 *       200:
 *         description: Home location set successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 message:
 *                   type: string
 *                 homeLocation:
 *                   type: object
 *       400:
 *         description: Invalid input or missing required fields
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Server error while setting home location
 *
 *   get:
 *     summary: Get user's home location
 *     description: Returns the current home location of the authenticated user.
 *     tags:
 *       - User
 *     responses:
 *       200:
 *         description: Successfully fetched home location
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 coordinates:
 *                   type: object
 *                   nullable: true
 *                   properties:
 *                     lat:
 *                       type: number
 *                     lng:
 *                       type: number
 *                     address:
 *                       type: string
 *                 homeLocation:
 *                   type: object
 *                   nullable: true
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: User not found
 *       500:
 *         description: Failed to fetch home location
 */
