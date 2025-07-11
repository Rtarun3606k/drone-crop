"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import { signIn, signOut } from "../lib/auth-client";
import { useSession } from "next-auth/react";
import { FiHome, FiBriefcase, FiInfo, FiMail, FiInbox } from "react-icons/fi";
import { useTranslations } from "../lib/client-i18n";
import LanguageSwitcher from "./LanguageSwitcher";

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const { data: session, status } = useSession();
  const isLoading = status === "loading";
  // Get translations and make sure they're rendered only once
  const t = useTranslations("common");

  // Toggle mobile menu
  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <nav
      className={"fixed top-0 left-0 w-full z-50 transition-all duration-500 bg-black/50 backdrop-blur-sm"}
    >
      <style dangerouslySetInnerHTML={{
        __html: `
          @keyframes textShine {
            0% {
              background-position: 0% 50%;
            }
            100% {
              background-position: 100% 50%;
            }
          }
        `
      }} />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16 items-center">
          {/* Logo and brand name */}
          <div className="flex-shrink-0 flex items-center">
            <Link href="/" className="flex items-center space-x-2">
              <div className="w-10 h-10 rounded-full bg-green-600 flex items-center justify-center">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-6 w-6 text-white"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z"
                  />
                </svg>
              </div>
              <span
                className="text-xl font-bold cursor-default bg-clip-text text-transparent"
                style={{
                  background: 'linear-gradient(to right, #22c55e 20%, #ffffff 30%, #10b981 70%, #16a34a 80%)',
                  backgroundSize: '500% auto',
                  WebkitBackgroundClip: 'text',
                  backgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  animation: 'textShine 4s ease-in-out infinite alternate',
                  display: "inline-block",
                  minHeight: "28px",
                  position: "relative",
                  zIndex: 10,
                }}
              >
                {t("brand")}
              </span>
            </Link>
          </div>

          {/* Desktop navigation links */}
          <div className="hidden md:flex items-center space-x-8">
            <Link
              href="/home"
              className="flex items-center text-white hover:text-green-400 transition-colors space-x-2"
            >
              <FiHome className="inline-block mr-1" />
              <span> {t("home")}</span>
            </Link>
            {session && session.user ? (
              <Link
                href="/dashboard"
                className="flex items-center text-white hover:text-green-400 transition-colors space-x-2"
              >
                {/* <FiHome className="inline-block mr-1" /> */}
                <FiInbox className="inline-block mr-1" />
                <span> {t("dashboard")} </span>
                {/* <span> dashboard </span> */}
              </Link>
            ) : null}

            <Link
              href="/services"
              className="flex items-center text-white hover:text-green-400 transition-colors space-x-2"
            >
              <FiBriefcase className="inline-block mr-1" />
              <span>{t("services")}</span>
            </Link>
            <Link
              href="/about"
              className="flex items-center text-white hover:text-green-400 transition-colors space-x-2"
            >
              <FiInfo className="inline-block mr-1" />
              <span>{t("about")}</span>
            </Link>
            <Link
              href="/contact"
              className="flex items-center text-white hover:text-green-400 transition-colors space-x-2"
            >
              <FiMail className="inline-block mr-1" />
              <span>{t("contact")}</span>
            </Link>

            {/* Auth buttons based on session state */}
            {isLoading ? (
              <div className="h-8 w-20 bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
            ) : session ? (
              <div className="flex items-center space-x-6">
                <LanguageSwitcher />
                <div className="flex items-center ml-2">
                  {session.user?.image ? (
                    <Link href="/profile" className="flex items-center">
                      <Image
                        src={session.user.image}
                        alt={session.user.name || "User profile"}
                        width={32}
                        height={32}
                        className="rounded-full"
                      />
                    </Link>
                  ) : (
                    <div className="h-8 w-8 rounded-full bg-green-100 dark:bg-green-800 flex items-center justify-center">
                      <span className="text-green-700 dark:text-green-300 font-medium text-sm">
                        {session.user?.name?.charAt(0) ||
                          session.user?.email?.charAt(0) ||
                          "U"}
                      </span>
                    </div>
                  )}
                  <Link
                    href={"/profile"}
                    className="ml-2 text-sm text-gray-700 dark:text-gray-300"
                  >
                    {session.user?.name?.split(" ")[0] ||
                      session.user?.email?.split("@")[0] ||
                      "User"}
                  </Link>
                </div>
                <button
                  onClick={() => signOut({ redirect: true, callbackUrl: "/" })}
                  className="text-sm px-3 py-1 bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900 dark:text-red-100 dark:hover:bg-red-800 rounded-full transition-colors"
                >
                  {t("signout")}
                </button>
              </div>
            ) : (
              <div className="flex space-x-6 items-center">
                <LanguageSwitcher />
                <Link
                  href="/login"
                  className="text-sm px-4 py-2 border border-green-500 text-green-600 hover:bg-green-50 dark:text-green-400 dark:hover:bg-green-900/30 rounded-full transition-colors"
                >
                  {t("login")}
                </Link>
                <button
                  onClick={() => signIn("google")}
                  className="text-sm px-4 py-2 bg-green-600 text-white hover:bg-green-700 rounded-full transition-colors"
                >
                  {t("signup")}
                </button>
              </div>
            )}
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={toggleMenu}
              className="inline-flex items-center justify-center p-2 rounded-md text-gray-700 dark:text-gray-300 hover:text-green-600 dark:hover:text-green-400 focus:outline-none"
              aria-expanded={isMenuOpen}
            >
              <span className="sr-only">
                {isMenuOpen ? t("closeMenu") : t("openMenu")}
              </span>
              {isMenuOpen ? (
                <svg
                  className="h-6 w-6"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              ) : (
                <svg
                  className="h-6 w-6"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      <div
        className={`${
          isMenuOpen ? "block" : "hidden"
        } md:hidden bg-white dark:bg-gray-900 shadow-lg`}
      >
        <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
          <Link
            href="/"
            className="flex items-center px-3 py-2 rounded-md text-base font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
            onClick={() => setIsMenuOpen(false)}
          >
            <FiHome className="inline-block mr-2" />
            {t("home")}
          </Link>
          <Link
            href="/services"
            className="flex items-center px-3 py-2 rounded-md text-base font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
            onClick={() => setIsMenuOpen(false)}
          >
            <FiBriefcase className="inline-block mr-2" />
            {t("services")}
          </Link>
          <Link
            href="/about"
            className="flex items-center px-3 py-2 rounded-md text-base font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
            onClick={() => setIsMenuOpen(false)}
          >
            <FiInfo className="inline-block mr-2" />
            {t("about")}
          </Link>
          <Link
            href="/contact"
            className="flex items-center px-3 py-2 rounded-md text-base font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
            onClick={() => setIsMenuOpen(false)}
          >
            <FiMail className="inline-block mr-2" />
            {t("contact")}
          </Link>
        </div>
        <div className="pt-4 pb-3 border-t border-gray-200 dark:border-gray-700">
          {isLoading ? (
            <div className="px-4">
              <div className="h-8 w-24 bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
            </div>
          ) : session ? (
            <div className="px-4">
              <div className="mb-4">
                <LanguageSwitcher />
              </div>
              <div className="flex items-center py-2">
                {session.user?.image ? (
                  <Image
                    src={session.user.image}
                    alt={session.user.name || "User profile"}
                    width={32}
                    height={32}
                    className="rounded-full"
                  />
                ) : (
                  <div className="h-8 w-8 rounded-full bg-green-100 dark:bg-green-800 flex items-center justify-center">
                    <span className="text-green-700 dark:text-green-300 font-medium">
                      {session.user?.name?.charAt(0) ||
                        session.user?.email?.charAt(0) ||
                        "U"}
                    </span>
                  </div>
                )}
                <Link
                  href={"/profile"}
                  className="ml-3 text-base font-medium text-gray-800 dark:text-gray-200"
                >
                  {session.user?.name || session.user?.email || "User"}
                </Link>
              </div>
              <div className="mt-3 space-y-1">
                <Link
                  href="/profile"
                  className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => setIsMenuOpen(false)}
                >
                  {t("profile")}
                </Link>
                <Link
                  href="/dashboard"
                  className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => setIsMenuOpen(false)}
                >
                  {t("dashboard")}
                </Link>
                <button
                  onClick={() => {
                    signOut({ redirect: true, callbackUrl: "/" });
                    setIsMenuOpen(false);
                  }}
                  className="block w-full text-left px-3 py-2 rounded-md text-base font-medium text-red-600 dark:text-red-400 hover:bg-gray-100 dark:hover:bg-gray-800"
                >
                  {t("signout")}
                </button>
              </div>
            </div>
          ) : (
            <div className="px-4 py-2 space-y-4">
              <div className="mb-2">
                <LanguageSwitcher />
              </div>
              <Link
                href="/login"
                className="block text-center px-4 py-2 border border-green-500 text-green-600 hover:bg-green-50 dark:text-green-400 dark:hover:bg-green-900/30 rounded-md"
                onClick={() => setIsMenuOpen(false)}
              >
                {t("login")}
              </Link>
              <button
                onClick={() => {
                  signIn("google");
                  setIsMenuOpen(false);
                }}
                className="block w-full px-4 py-2 bg-green-600 text-white hover:bg-green-700 rounded-md"
              >
                {t("signup")}
              </button>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
