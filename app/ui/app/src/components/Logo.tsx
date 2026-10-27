import mark from "../assets/wildllama-mark.svg";
import glyph from "../assets/wildllama-glyph.svg";

interface LogoProps {
  size?: number;
  containerClassName?: string;
  showBackground?: boolean;
}

export default function Logo({
  size = 60,
  containerClassName = "mb-8",
  showBackground = true,
}: LogoProps = {}) {
  return (
    <div className={`flex justify-center select-none ${containerClassName}`}>
      <img
        src={showBackground ? mark : glyph}
        alt="WildLlama"
        width={size}
        height={size}
        draggable={false}
        className={showBackground ? "select-none" : "select-none dark:invert"}
      />
    </div>
  );
}
