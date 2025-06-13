"use client";

import ClientSessionProvider from "@/components/ClientSessionProvider";

import { useState } from "react";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const images = [
    "Nature Cube.jpg",
    "boliviainteligente-TFdUQaIjV6s-unsplash.jpg",
    "hassaan-here-bKfkhVRAJTQ-unsplash.jpg",
    "igor-omilaev-6-Y_Hxoh7VU-unsplash.jpg",
    "jigar-panchal-bikKtf6mgZ8-unsplash.jpg",
    "maxim-berg-WG56pNhliUQ-unsplash.jpg",
    "pexels-alexander-grey-3679453.jpg",
    "pexels-suket-dedhia-570026.jpg",
    "pexels-elijahsad-3473569.jpg",
  ];

  const [image, setImage] = useState(0);
  // useEffect(() => {
  //   // setInterval(() => {
  //   //   setImage((prev) => (prev + 1) % images.length);
  //   // }, 2000);
  // }, []);
  return (
    <div className="     ">
      <ClientSessionProvider>
        {children}
        
        <img
          className="fixed top-0 bottom-0 h-full object-cover -z-50"
          src={`/Wallpapers/${images[image]}`}
          height={4000}
          width={4000}
          alt={"wallpaper"}
        />
      </ClientSessionProvider>
    </div>
  );
}
