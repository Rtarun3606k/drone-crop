import { auth } from "@/app/auth";
import { prisma } from "@/app/lib/prisma-server";
import { NextResponse } from "next/server";

// Map locale codes to Language enum values
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

// Map Language enum back to locale codes
const mapLanguageEnumToLocale = (languageEnum) => {
  const mapping = {
    En: "en",
    Ta: "ta",
    Hi: "hi",
    Te: "te",
    Ml: "ml",
    Kn: "kn",
  };
  return mapping[languageEnum] || "en";
};

export async function GET(request) {
  try {
    const session = await auth();

    // Check authentication
    if (!session?.user) {
      return NextResponse.json(
        { error: "Unauthorized. Please log in." },
        { status: 401 }
      );
    }

    // Get user's language preference
    const user = await prisma.user.findUnique({
      where: { id: session.user.id },
      select: { defaultLanguage: true },
    });

    if (!user) {
      return NextResponse.json({ error: "User not found." }, { status: 404 });
    }

    const localeCode = mapLanguageEnumToLocale(user.defaultLanguage);

    return NextResponse.json(
      {
        success: true,
        defaultLanguage: user.defaultLanguage,
        locale: localeCode,
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Error fetching language preference:", error);
    return NextResponse.json(
      { error: "Failed to fetch language preference." },
      { status: 500 }
    );
  }
}

export async function POST(request) {
  try {
    const session = await auth();

    // Check authentication
    if (!session?.user) {
      return NextResponse.json(
        { error: "Unauthorized. Please log in." },
        { status: 401 }
      );
    }

    const { locale } = await request.json();

    if (!locale) {
      return NextResponse.json(
        { error: "Locale is required." },
        { status: 400 }
      );
    }

    // Map locale to Language enum
    const languageEnum = mapLocaleToLanguageEnum(locale);

    // Update user's default language
    const updatedUser = await prisma.user.update({
      where: { id: session.user.id },
      data: { defaultLanguage: languageEnum },
    });

    return NextResponse.json(
      {
        success: true,
        message: "Language preference updated successfully",
        defaultLanguage: updatedUser.defaultLanguage,
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Error updating language preference:", error);
    return NextResponse.json(
      { error: "Failed to update language preference." },
      { status: 500 }
    );
  }
}
