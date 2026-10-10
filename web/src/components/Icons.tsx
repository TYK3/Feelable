// Small inline stroke icons (24px grid, currentColor). Decorative: aria-hidden.
import type { SVGProps } from "react";

type P = SVGProps<SVGSVGElement> & { size?: number };

function Svg({ size = 20, children, ...rest }: P) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const IconUpload = (p: P) => (
  <Svg {...p}><path d="M12 16V4" /><path d="m6 10 6-6 6 6" /><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" /></Svg>
);
export const IconCamera = (p: P) => (
  <Svg {...p}><path d="M4 8h3l2-3h6l2 3h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1Z" /><circle cx="12" cy="13.5" r="3.5" /></Svg>
);
export const IconImage = (p: P) => (
  <Svg {...p}><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="9" cy="10" r="2" /><path d="m21 16-5-5-9 9" /></Svg>
);
export const IconDownload = (p: P) => (
  <Svg {...p}><path d="M12 4v12" /><path d="m6 10 6 6 6-6" /><path d="M5 20h14" /></Svg>
);
export const IconHand = (p: P) => (
  <Svg {...p}><path d="M8 13V5.5a1.5 1.5 0 0 1 3 0V11" /><path d="M11 10.5V4.5a1.5 1.5 0 0 1 3 0v6" /><path d="M14 10.5v-4a1.5 1.5 0 0 1 3 0V14a6 6 0 0 1-6 6h-.5a6 6 0 0 1-4.6-2.2L3.6 15a1.5 1.5 0 0 1 2.3-1.9L8 15.5" /></Svg>
);
export const IconLink = (p: P) => (
  <Svg {...p}><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1" /><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1" /></Svg>
);
export const IconCheck = (p: P) => (
  <Svg {...p}><path d="m5 12.5 4.5 4.5L19 7.5" /></Svg>
);
export const IconWarn = (p: P) => (
  <Svg {...p}><path d="M12 4 2.8 19.5h18.4Z" /><path d="M12 10v4" /><path d="M12 17h.01" /></Svg>
);
export const IconX = (p: P) => (
  <Svg {...p}><path d="M6 6l12 12M18 6 6 18" /></Svg>
);
export const IconArrowLeft = (p: P) => (
  <Svg {...p}><path d="M19 12H5" /><path d="m11 18-6-6 6-6" /></Svg>
);
export const IconArrowRight = (p: P) => (
  <Svg {...p}><path d="M5 12h14" /><path d="m13 6 6 6-6 6" /></Svg>
);
export const IconZoom = (p: P) => (
  <Svg {...p}><circle cx="11" cy="11" r="6.5" /><path d="m20 20-4.2-4.2" /><path d="M11 8.5v5M8.5 11h5" /></Svg>
);
export const IconSparkle = (p: P) => (
  <Svg {...p}><path d="M12 3v4M12 17v4M3 12h4M17 12h4" /><path d="m6.3 6.3 2.1 2.1M15.6 15.6l2.1 2.1M6.3 17.7l2.1-2.1M15.6 8.4l2.1-2.1" /></Svg>
);
export const IconClock = (p: P) => (
  <Svg {...p}><circle cx="12" cy="12" r="8.5" /><path d="M12 7.5V12l3 2" /></Svg>
);
export const IconPencil = (p: P) => (
  <Svg {...p}><path d="M4 20h4L19 9a2.8 2.8 0 0 0-4-4L4 16Z" /><path d="m13.5 6.5 4 4" /></Svg>
);
export const IconRefresh = (p: P) => (
  <Svg {...p}><path d="M20 11a8 8 0 0 0-14.6-4.5L4 8" /><path d="M4 4v4h4" /><path d="M4 13a8 8 0 0 0 14.6 4.5L20 16" /><path d="M20 20v-4h-4" /></Svg>
);
export const IconCube = (p: P) => (
  <Svg {...p}><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9Z" /><path d="m4 7.5 8 4.5 8-4.5" /><path d="M12 12v9" /></Svg>
);
export const IconClipboard = (p: P) => (
  <Svg {...p}><rect x="6" y="4" width="12" height="17" rx="2" /><path d="M9 4V3h6v1" /><path d="M9 10h6M9 14h4" /></Svg>
);
export const IconSpeaker = (p: P) => (
  <Svg {...p}><path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4Z" /><path d="M15.5 9a4 4 0 0 1 0 6" /><path d="M18 6.5a7.5 7.5 0 0 1 0 11" /></Svg>
);
export const IconSpeakerOff = (p: P) => (
  <Svg {...p}><path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4Z" /><path d="m16 10 4 4M20 10l-4 4" /></Svg>
);
export const IconShare = (p: P) => (
  <Svg {...p}><path d="M12 15V3.5" /><path d="m7.5 8 4.5-4.5L16.5 8" /><path d="M8 11H6a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-8a1 1 0 0 0-1-1h-2" /></Svg>
);
export const IconEye = (p: P) => (
  <Svg {...p}><path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z" /><circle cx="12" cy="12" r="3" /></Svg>
);
export const IconChevronDown = (p: P) => (
  <Svg {...p}><path d="m6 9 6 6 6-6" /></Svg>
);
export const IconSplit = (p: P) => (
  <Svg {...p}><circle cx="6.5" cy="12" r="3.5" /><circle cx="17.5" cy="12" r="3.5" /><path d="M12 4v16" strokeDasharray="2 2.5" /></Svg>
);
export const IconMerge = (p: P) => (
  <Svg {...p}><path d="M5 5c0 6 7 5 7 11v4" /><path d="M19 5c0 6-7 5-7 11" /></Svg>
);
export const IconLock = (p: P) => (
  <Svg {...p}><rect x="5" y="10.5" width="14" height="10" rx="2" /><path d="M8 10.5V7.5a4 4 0 0 1 8 0v3" /></Svg>
);
export const IconPlus = (p: P) => (
  <Svg {...p}><path d="M12 5v14M5 12h14" /></Svg>
);
export const IconMinus = (p: P) => (
  <Svg {...p}><path d="M5 12h14" /></Svg>
);
export const IconCopy = (p: P) => (
  <Svg {...p}><rect x="8.5" y="8.5" width="12" height="12" rx="2" /><path d="M15.5 8.5V5a1.5 1.5 0 0 0-1.5-1.5H5A1.5 1.5 0 0 0 3.5 5v9A1.5 1.5 0 0 0 5 15.5h3.5" /></Svg>
);
export const IconPrinter = (p: P) => (
  <Svg {...p}><path d="M7 9V3.5h10V9" /><rect x="3.5" y="9" width="17" height="8" rx="2" /><path d="M7 14h10v6.5H7Z" /></Svg>
);
