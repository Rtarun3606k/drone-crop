"use client";

import { useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useParams } from "next/navigation";
import { FiGlobe } from "react-icons/fi";

const LanguageSwitcher = () => {
  const router = useRouter();
  const pathname = usePathname();
  const params = useParams();
  const currentLocale = params.locale || "en";

  const [isOpen, setIsOpen] = useState(false);

  const languages = [
    { code: "en", name: "English" },
    { code: "ka", name: "ಕನ್ನಡ" }, // Kannada
    { code: "hi", name: "हिंदी" }, // Hindi
    { code: "te", name: "తెలుగు" }, // Telugu
    { code: "ta", name: "தமிழ்" }, // Tamil
    { code: "ml", name: "മലയാളം" }, // Malayalam
  ];

  const handleLanguageChange = (locale) => {
    setIsOpen(false);

    // Get the current path, replace the locale part
    const currentPath = pathname || "";
    const pathWithoutLocale = currentPath.replace(/^\/[^\/]+/, "");

    // Navigate to the same page but with different locale
    const newPath = `/${locale}${pathWithoutLocale}`;
    router.push(newPath);
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-1 text-gray-700 dark:text-gray-300 hover:text-green-600 dark:hover:text-green-400 px-2 py-1 rounded-full border border-gray-300 dark:border-gray-600 hover:border-green-500 dark:hover:border-green-500 transition-colors"
      >
        <FiGlobe className="h-3.5 w-3.5" />
        <span className="text-xs font-medium">
          {currentLocale.toUpperCase()}
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
