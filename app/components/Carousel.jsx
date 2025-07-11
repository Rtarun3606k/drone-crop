import React from "react";
import { useTranslations } from "next-intl";

export const Carousel = () => {
  const t = useTranslations("Carousel");
  const items = t.raw("items");
  const backgrounds = [
    "/carouselImages/carousel1.png",
    "/carouselImages/carousel2.png",
    "/carouselImages/carousel3.png",
    "/carouselImages/carousel4.png",
  ];

  return (
    <div className="w-full px-4 sm:px-6 lg:px-8">
      <h1 className="text-center text-2xl sm:text-3xl lg:text-4xl font-bold mb-6 sm:mb-8 lg:mb-9">{t("title")}</h1>
      <div className="carousel carousel-center rounded-box w-full overflow-x-auto">
        {items.map((item, idx) => (
          <div
            key={item.title}
            className="carousel-item flex flex-col items-center justify-center p-3 sm:p-4 lg:p-6 relative min-h-[250px] sm:min-h-[300px] lg:min-h-[350px] w-full sm:w-[90%] md:w-[80%] lg:w-[70%] xl:max-w-xl mx-auto rounded-lg mb-6 sm:mb-8 lg:mb-10"
            style={{
              backgroundImage: `url(${backgrounds[idx]})`,
              backgroundSize: "cover",
              backgroundPosition: "center",
            }}
          >
            <div className="absolute inset-0 w-full h-full backdrop-blur-sm bg-black/30 z-0 rounded-lg" />
            <div className="relative z-10 flex flex-col items-center justify-center w-full h-full text-white p-3 sm:p-4 lg:p-6">
              <h2 className="text-lg sm:text-xl lg:text-2xl font-bold mb-2 sm:mb-3 lg:mb-4 text-center">{item.title}</h2>
              <ul className="list-disc list-inside text-left text-sm sm:text-base w-full max-w-xs sm:max-w-md lg:max-w-lg mx-auto">
                {item.features.map((feature, i) => (
                  <li key={i} className="mb-1 sm:mb-2">{feature}</li>
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
