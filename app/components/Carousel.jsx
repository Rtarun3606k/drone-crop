
import React from "react";
import { useTranslations } from "next-intl";

export const Carousel = () => {
  const t = useTranslations("Carousel");
  const items = t.raw("items");
  const backgrounds = [
    // "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80",
    "/carouselImages/carousel1.png",
    // "https://images.unsplash.com/photo-1464983953574-0892a716854b?auto=format&fit=crop&w=800&q=80",
    // "https://images.unsplash.com/photo-1501785888041-af3ef285b470?auto=format&fit=crop&w=800&q=80",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSj8n5deEHPTsYoYG0CmhyWoOYOqaNw1Kpf-Q&s",
    "/carouselImages/carousel3.png",
    // "https://images.unsplash.com/photo-1465101046530-73398c7f28ca?auto=format&fit=crop&w=800&q=80"
    "https://t4.ftcdn.net/jpg/03/27/36/95/360_F_327369570_CAxxxHHLvjk6IJ3wGi1kuW6WTtqjaMpc.jpg"
  ];

  return (
    <div>
      <h1 className="text-center text-4xl font-bold mb-9">{t("title")}</h1>
      <div className="carousel carousel-center rounded-box">
        {items.map((item, idx) => (
          <div
            key={item.title}
            className="carousel-item flex flex-col items-center justify-center p-4 relative min-h-[300px] w-full max-w-xl mx-auto rounded-lg mb-10"
            style={{
              backgroundImage: `url(${backgrounds[idx]})`,
              backgroundSize: "cover",
              backgroundPosition: "center",
            }}
          >
            <div className="absolute inset-0 w-full h-full backdrop-blur-sm bg-black/30 z-0 rounded-lg" />
            <div className="relative z-10 flex flex-col items-center justify-center w-full h-full text-white p-6">
              <h2 className="text-xl font-bold mb-2 text-center">{item.title}</h2>
              <ul className="list-disc list-inside text-left max-w-md mx-auto">
                {item.features.map((feature, i) => (
                  <li key={i}>{feature}</li>
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
