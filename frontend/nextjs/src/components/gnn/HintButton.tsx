"use client";
import React, { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

type Props = {
  content: React.ReactNode;
  placement?: "top" | "bottom" | "left" | "right";
  delayMs?: number;
  children?: React.ReactNode;
};

type PopupPos = { top: number; left: number; transform: string; dark: boolean };

const GAP = 8;
const SCREEN_PAD = 8;

function computePos(rect: DOMRect, placement: string, dark: boolean): PopupPos {
  switch (placement) {
    case "bottom":
      return { top: rect.bottom + GAP, left: rect.left + rect.width / 2, transform: "translateX(-50%)", dark };
    case "left":
      return { top: rect.top + rect.height / 2, left: rect.left - GAP, transform: "translate(-100%, -50%)", dark };
    case "right":
      return { top: rect.top + rect.height / 2, left: rect.right + GAP, transform: "translateY(-50%)", dark };
    default: // top
      return { top: rect.top - GAP, left: rect.left + rect.width / 2, transform: "translate(-50%, -100%)", dark };
  }
}

export default function HintButton({ content, placement = "top", delayMs = 300, children }: Props) {
  const [pos, setPos] = useState<PopupPos | null>(null);
  const showTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const btnRef = useRef<HTMLSpanElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);

  // Clamp popup into the viewport after it renders.
  // Direct DOM mutation avoids a re-render loop that setState would cause.
  useLayoutEffect(() => {
    const el = popupRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const dx = r.right > vw - SCREEN_PAD ? r.right - (vw - SCREEN_PAD)
             : r.left  < SCREEN_PAD      ? r.left - SCREEN_PAD
             : 0;
    const dy = r.bottom > vh - SCREEN_PAD ? r.bottom - (vh - SCREEN_PAD)
             : r.top    < SCREEN_PAD      ? r.top - SCREEN_PAD
             : 0;
    if (dx !== 0) el.style.left = `${parseFloat(el.style.left) - dx}px`;
    if (dy !== 0) el.style.top  = `${parseFloat(el.style.top)  - dy}px`;
  }, [pos]);

  const cancelHide = () => { if (hideTimer.current) clearTimeout(hideTimer.current); };
  const cancelShow = () => { if (showTimer.current) clearTimeout(showTimer.current); };

  const scheduleHide = () => {
    cancelHide();
    hideTimer.current = setTimeout(() => setPos(null), 120);
  };

  const showPopup = () => {
    cancelHide();
    if (!btnRef.current) return;
    const dark = document.getElementById("app")?.classList.contains("dark") ?? false;
    setPos(computePos(btnRef.current.getBoundingClientRect(), placement, dark));
  };

  const onBtnEnter = () => { cancelHide(); cancelShow(); showTimer.current = setTimeout(showPopup, delayMs); };
  const onBtnLeave = () => { cancelShow(); scheduleHide(); };

  const popup = pos ? (
    <div
      ref={popupRef}
      className={`hint-popup${pos.dark ? " hint-popup--dark" : ""}`}
      role="tooltip"
      style={{ position: "fixed", top: pos.top, left: pos.left, transform: pos.transform }}
      onMouseEnter={cancelHide}
      onMouseLeave={scheduleHide}
    >
      {content}
    </div>
  ) : null;

  return (
    <>
      <span
        ref={btnRef}
        role={children ? undefined : "img"}
        aria-label={children ? undefined : "Show help"}
        className={children ? undefined : "hint-btn"}
        style={children ? { display: "inline-flex" } : undefined}
        onMouseEnter={onBtnEnter}
        onMouseLeave={onBtnLeave}
        onFocus={children ? undefined : showPopup}
        onBlur={children ? undefined : scheduleHide}
        tabIndex={children ? undefined : 0}
      >
        {children ?? "?"}
      </span>
      {typeof document !== "undefined" ? createPortal(popup, document.body) : popup}
    </>
  );
}
