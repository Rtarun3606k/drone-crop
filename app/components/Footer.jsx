import React from "react";
import Link from "next/link";
import { FiHome, FiUser, FiBriefcase, FiInfo, FiMail, FiFacebook, FiTwitter, FiInstagram, FiLinkedin } from "react-icons/fi";
import { getTranslations } from "next-intl/server";

const Footer = async () => {
  const t = await getTranslations("common");

  return (
    <footer className={`relative ${'bg-gradient-to-t from-gray-900 to-black'} text-white mt-auto overflow-hidden`}>
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
      
      
      {/* Decorative gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-black via-transparent to-transparent" />
      
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-16">
        {/* Main footer content */}
        <div className="flex flex-col lg:flex-row lg:justify-between lg:items-center gap-12 mb-12">
          {/* Company name - Left */}
          <div className="text-center lg:text-left">
            <h2 
              className="text-3xl lg:text-4xl font-bold tracking-wide leading-normal cursor-default bg-clip-text text-transparent"
              style={{
                background: 'linear-gradient(to right, #22c55e 20%, #ffffff 30%, #10b981 70%, #16a34a 80%)',
                backgroundSize: '500% auto',
                WebkitBackgroundClip: 'text',
                backgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                animation: 'textShine 4s ease-in-out infinite alternate'
              }}
            >
              {t("brand")}
            </h2>
            {/* <div className="mt-2 w-20 h-1 bg-gradient-to-r from-green-400 to-green-600 rounded-full mx-auto lg:mx-0 animate-pulse" /> */}
          </div>

          {/* Navigation icons - Center */}
          <nav className="flex flex-wrap justify-center gap-8 lg:gap-12">
            <Link
              href="/"
              className="group flex flex-col items-center gap-3 text-gray-300 hover:text-green-400 transition-all duration-300 transform hover:scale-105 hover:-translate-y-1"
            >
              <div className="relative p-3 rounded-xl bg-gray-800/50 backdrop-blur-sm group-hover:bg-green-500/20 group-hover:shadow-lg group-hover:shadow-green-500/25 transition-all duration-300">
                <FiHome size={24} className="group-hover:scale-110 transition-transform duration-300" />
                <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-green-400/0 to-green-600/0 group-hover:from-green-400/10 group-hover:to-green-600/10 transition-all duration-300" />
              </div>
              <span className="text-sm font-medium tracking-wide">{t("home")}</span>
            </Link>
            <Link
              href="/profile"
              className="group flex flex-col items-center gap-3 text-gray-300 hover:text-green-400 transition-all duration-300 transform hover:scale-105 hover:-translate-y-1"
            >
              <div className="relative p-3 rounded-xl bg-gray-800/50 backdrop-blur-sm group-hover:bg-green-500/20 group-hover:shadow-lg group-hover:shadow-green-500/25 transition-all duration-300">
                <FiUser size={24} className="group-hover:scale-110 transition-transform duration-300" />
                <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-green-400/0 to-green-600/0 group-hover:from-green-400/10 group-hover:to-green-600/10 transition-all duration-300" />
              </div>
              <span className="text-sm font-medium tracking-wide">{t("profile")}</span>
            </Link>
            <Link
              href="/services"
              className="group flex flex-col items-center gap-3 text-gray-300 hover:text-green-400 transition-all duration-300 transform hover:scale-105 hover:-translate-y-1"
            >
              <div className="relative p-3 rounded-xl bg-gray-800/50 backdrop-blur-sm group-hover:bg-green-500/20 group-hover:shadow-lg group-hover:shadow-green-500/25 transition-all duration-300">
                <FiBriefcase size={24} className="group-hover:scale-110 transition-transform duration-300" />
                <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-green-400/0 to-green-600/0 group-hover:from-green-400/10 group-hover:to-green-600/10 transition-all duration-300" />
              </div>
              <span className="text-sm font-medium tracking-wide">{t("services")}</span>
            </Link>
            <Link
              href="/about"
              className="group flex flex-col items-center gap-3 text-gray-300 hover:text-green-400 transition-all duration-300 transform hover:scale-105 hover:-translate-y-1"
            >
              <div className="relative p-3 rounded-xl bg-gray-800/50 backdrop-blur-sm group-hover:bg-green-500/20 group-hover:shadow-lg group-hover:shadow-green-500/25 transition-all duration-300">
                <FiInfo size={24} className="group-hover:scale-110 transition-transform duration-300" />
                <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-green-400/0 to-green-600/0 group-hover:from-green-400/10 group-hover:to-green-600/10 transition-all duration-300" />
              </div>
              <span className="text-sm font-medium tracking-wide">{t("about")}</span>
            </Link>
            <Link
              href="/contact"
              className="group flex flex-col items-center gap-3 text-gray-300 hover:text-green-400 transition-all duration-300 transform hover:scale-105 hover:-translate-y-1"
            >
              <div className="relative p-3 rounded-xl bg-gray-800/50 backdrop-blur-sm group-hover:bg-green-500/20 group-hover:shadow-lg group-hover:shadow-green-500/25 transition-all duration-300">
                <FiMail size={24} className="group-hover:scale-110 transition-transform duration-300" />
                <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-green-400/0 to-green-600/0 group-hover:from-green-400/10 group-hover:to-green-600/10 transition-all duration-300" />
              </div>
              <span className="text-sm font-medium tracking-wide">{t("contact")}</span>
            </Link>
          </nav>

          {/* Social icons - Right */}
          <div className="flex justify-center lg:justify-end gap-4">
            <a
              href="https://facebook.com"
              target="_blank"
              rel="noopener noreferrer"
              className="group relative p-3 rounded-full bg-gray-800/60 backdrop-blur-sm hover:bg-green-500/90 transition-all duration-300 transform hover:scale-110 hover:rotate-6 hover:shadow-lg hover:shadow-green-500/25"
            >
              <FiFacebook size={20} className="group-hover:scale-110 transition-transform duration-300" />
              <div className="absolute inset-0 rounded-full bg-gradient-to-r from-green-400/0 to-green-600/0 group-hover:from-green-400/20 group-hover:to-green-600/20 transition-all duration-300" />
            </a>
            <a
              href="https://twitter.com"
              target="_blank"
              rel="noopener noreferrer"
              className="group relative p-3 rounded-full bg-gray-800/60 backdrop-blur-sm hover:bg-green-500/90 transition-all duration-300 transform hover:scale-110 hover:rotate-6 hover:shadow-lg hover:shadow-green-500/25"
            >
              <FiTwitter size={20} className="group-hover:scale-110 transition-transform duration-300" />
              <div className="absolute inset-0 rounded-full bg-gradient-to-r from-green-400/0 to-green-600/0 group-hover:from-green-400/20 group-hover:to-green-600/20 transition-all duration-300" />
            </a>
            <a
              href="https://instagram.com"
              target="_blank"
              rel="noopener noreferrer"
              className="group relative p-3 rounded-full bg-gray-800/60 backdrop-blur-sm hover:bg-green-500/90 transition-all duration-300 transform hover:scale-110 hover:rotate-6 hover:shadow-lg hover:shadow-green-500/25"
            >
              <FiInstagram size={20} className="group-hover:scale-110 transition-transform duration-300" />
              <div className="absolute inset-0 rounded-full bg-gradient-to-r from-green-400/0 to-green-600/0 group-hover:from-green-400/20 group-hover:to-green-600/20 transition-all duration-300" />
            </a>
            <a
              href="https://linkedin.com"
              target="_blank"
              rel="noopener noreferrer"
              className="group relative p-3 rounded-full bg-gray-800/60 backdrop-blur-sm hover:bg-green-500/90 transition-all duration-300 transform hover:scale-110 hover:rotate-6 hover:shadow-lg hover:shadow-green-500/25"
            >
              <FiLinkedin size={20} className="group-hover:scale-110 transition-transform duration-300" />
              <div className="absolute inset-0 rounded-full bg-gradient-to-r from-green-400/0 to-green-600/0 group-hover:from-green-400/20 group-hover:to-green-600/20 transition-all duration-300" />
            </a>
          </div>
        </div>

        {/* Copyright info - Bottom centered */}
        <div className="border-t border-gray-700/50 pt-8">
          <div className="text-center space-y-4">
            <div className="flex flex-col sm:flex-row items-center justify-center gap-2 sm:gap-6 text-sm text-gray-300">
              <div className="flex items-center gap-2">
                <span>{t("contact")}:</span>
                <a 
                  href="mailto:info@dronecrop.com" 
                  className="text-green-400 hover:text-green-300 transition-colors duration-300 hover:underline decoration-2 underline-offset-2"
                >
                  info@dronecrop.com
                </a>
              </div>
              <div className="hidden sm:block text-gray-500">|</div>
              <div className="text-green-400 font-medium">+1 (555) 123-4567</div>
            </div>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-2 sm:gap-4 text-sm text-gray-400">
              <span>&copy; {new Date().getFullYear()} {t("brand")}. {t("rights")}</span>
              <div className="hidden sm:block text-gray-600">•</div>
              <span className="text-xs text-gray-500">Designed with care for the future</span>
            </div>
          </div>
        </div>
      </div>

      {/* Subtle background elements */}
      <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-green-500/50 to-transparent" />
    </footer>
  );
};

export default Footer;
