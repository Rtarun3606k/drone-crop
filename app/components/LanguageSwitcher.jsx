"use client";

import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useParams } from "next/navigation";
import { useSession } from "next-auth/react";
import { FiGlobe } from "react-icons/fi";

const LanguageSwitcher = () => {
  const router = useRouter();
  const pathname = usePathname();
  const params = useParams();
  const { data: session } = useSession();
  const currentLocale = params.locale || "en";

  const [isOpen, setIsOpen] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  const [isAutoSetting, setIsAutoSetting] = useState(false);

  // Auto-set language when user logs in (optimized to prevent infinite loops)
  useEffect(() => {
    const autoSetUserLanguage = async () => {
      // Only proceed if user is authenticated and we're not already updating
      if (!session?.user?.id || isUpdating || isAutoSetting) {
        return;
      }

      // Create a unique key for this session and current locale combination
      const autoSetKey = `autoSet_${session.user.id}_${currentLocale}`;

      // Check if we've already attempted auto-setting for this user in this session
      if (typeof window !== "undefined" && sessionStorage.getItem(autoSetKey)) {
        return;
      }

      setIsAutoSetting(true);

      try {
        // Fetch user's preferred language from the database
        const response = await fetch("/api/user/language", {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        });

        if (response.ok) {
          const data = await response.json();
          const userPreferredLocale = data.locale;

          // If the current locale doesn't match user's preference, redirect
          if (userPreferredLocale && userPreferredLocale !== currentLocale) {
            const currentPath = pathname || "";
            const pathWithoutLocale = currentPath.replace(/^\/[^\/]+/, "");
            const newPath = `/${userPreferredLocale}${pathWithoutLocale}`;

            console.log(
              `Auto-setting language to user preference: ${userPreferredLocale}`
            );

            // Mark as auto-set BEFORE redirect to prevent loops
            if (typeof window !== "undefined") {
              sessionStorage.setItem(autoSetKey, "true");
            }

            router.push(newPath);
            return; // Exit early to prevent setting the flag again
          }

          // Mark as auto-set even if no redirect was needed
          if (typeof window !== "undefined") {
            sessionStorage.setItem(autoSetKey, "true");
          }
        }
      } catch (error) {
        console.warn("Failed to auto-set user language preference:", error);
      } finally {
        setIsAutoSetting(false);
      }
    };

    // Only run when session changes from loading to authenticated
    if (session?.user?.id && !isUpdating && !isAutoSetting) {
      autoSetUserLanguage();
    }
  }, [session?.user?.id]); // Only depend on user ID to prevent dependency loops

  const languages = [
    { code: "en", name: "English" },
    { code: "kn", name: "ಕನ್ನಡ" }, // Kannada
    { code: "hi", name: "हिंदी" }, // Hindi
    { code: "te", name: "తెలుగు" }, // Telugu
    { code: "ta", name: "தமிழ்" }, // Tamil
    { code: "ml", name: "മലയാളം" }, // Malayalam
  ];

  const handleLanguageChange = async (locale) => {
    setIsOpen(false);
    setIsUpdating(true);

    try {
      // Get the current path, replace the locale part
      const currentPath = pathname || "";
      const pathWithoutLocale = currentPath.replace(/^\/[^\/]+/, "");

      // Navigate to the same page but with different locale
      const newPath = `/${locale}${pathWithoutLocale}`;

      // Update user's language preference if authenticated
      if (session?.user) {
        try {
          const response = await fetch("/api/user/language", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ locale }),
          });

          if (!response.ok) {
            console.warn("Failed to update language preference in database");
          }
        } catch (error) {
          console.warn("Error updating language preference:", error);
          // Don't block navigation if language update fails
        }
      }

      // Navigate regardless of whether the user is authenticated or DB update succeeded
      router.push(newPath);
    } catch (error) {
      console.error("Error changing language:", error);
    } finally {
      setIsUpdating(false);
    }
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={isUpdating || isAutoSetting}
        className={`flex items-center space-x-1 text-gray-700 dark:text-gray-300 hover:text-green-600 dark:hover:text-green-400 px-2 py-1 rounded-full border border-gray-300 dark:border-gray-600 hover:border-green-500 dark:hover:border-green-500 transition-colors ${
          isUpdating || isAutoSetting ? "opacity-50 cursor-not-allowed" : ""
        }`}
      >
        <FiGlobe
          className={`h-3.5 w-3.5 ${
            isUpdating || isAutoSetting ? "animate-spin" : ""
          }`}
        />
        <span className="text-xs font-medium">
          {isAutoSetting
            ? "Auto-setting..."
            : isUpdating
            ? "Updating..."
            : currentLocale.toUpperCase()}
        </span>
      </button>

      {isOpen && (
        <div
          className="absolute top-full mt-1 right-0 bg-white dark:bg-gray-800 rounded-md shadow-lg py-1 z-50 min-w-[120px] border border-gray-200 dark:border-gray-700"
          style={{ minWidth: "120px", maxWidth: "150px" }}
        >
          {languages.map((language) => (
            <button
              key={language.code}
              onClick={() => handleLanguageChange(language.code)}
              className={`block w-full text-left px-4 py-2 text-sm ${
                currentLocale === language.code
                  ? "bg-green-50 dark:bg-green-900/30 text-green-600 dark:text-green-400"
                  : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
              }`}
            >
              {language.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default LanguageSwitcher;
