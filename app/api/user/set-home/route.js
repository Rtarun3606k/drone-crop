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
