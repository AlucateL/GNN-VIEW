"use client";

import React, { useState, useEffect, useMemo, useRef } from "react";
import { NODE_GRID, NODE_MESH, EDGE_ENCODER, EDGE_DECODER, EDGE_MESH, ACCENT, TEXT, TEXT_MUTED, TEXT_DIM, palette } from '../../lib/colors';
import HintButton from "./HintButton";
import { HINT_GRID_NODES } from "../../lib/hints";

// ─────────────────────────────────────────────────────────────────────────────
// Palette — sourced from src/lib/colors.ts (single source of truth).
// ─────────────────────────────────────────────────────────────────────────────
const C = {
  g2m: EDGE_ENCODER, m2m: EDGE_MESH, m2g: EDGE_DECODER,
  grid: NODE_GRID, mesh: NODE_MESH,
  bgPanel: palette.ui.panelBackground,
  bgRail: 'rgba(0,0,0,0.025)',
  bgTint: 'rgba(0,0,0,0.03)',
  border: palette.ui.border,
  borderL: palette.ui.borderSubtle,
  text: TEXT,
  muted: TEXT_MUTED,
  dim: TEXT_DIM,
  accent: ACCENT,
};

const SANS = "'SF Pro Display','SF Pro Text',-apple-system,BlinkMacSystemFont,'Helvetica Neue',Arial,sans-serif";
const MONO = '"SF Mono","JetBrains Mono","Menlo",ui-monospace,monospace';

// ─────────────────────────────────────────────────────────────────────────────
// Level helpers
// ─────────────────────────────────────────────────────────────────────────────
export const MESH_STROKE_COARSE = 3.5;  // tunable: stroke weight for the coarsest mesh level
export const MESH_STROKE_FINE   = 0.8;  // tunable: stroke weight for the finest mesh level

export const MESH_NODE_COARSE_BOOST = 3.5;  // tunable: size multiplier for coarsest level
export const MESH_NODE_FINE_BOOST   = 0.8;  // tunable: size multiplier for finest level

export function nodeLevelBoost(type: string, L: number): number {
  if (L <= 1) return 1.0;
  const m = type.match(/^mesh_(\d+)$/);
  if (!m) return 1.0;
  const lvl = parseInt(m[1], 10);
  return MESH_NODE_FINE_BOOST + (MESH_NODE_COARSE_BOOST - MESH_NODE_FINE_BOOST) * lvl / (L - 1);
}

function levelWeight(level: number, L: number): number {
  if (L <= 1) return (MESH_STROKE_COARSE + MESH_STROKE_FINE) / 2;
  return MESH_STROKE_COARSE - (MESH_STROKE_COARSE - MESH_STROKE_FINE) * level / (L - 1);
}

function levelDash(level: number): string | null {
  return ([null, '3 2', '1.5 2', '1 1.5'] as (string | null)[])[Math.min(level, 3)] ?? null;
}

// strokeForLevel: coarser levels get heavier strokes (level 0 = finest = lightest)
// In the diagram, lvl is the actual mesh level index. strokeForLevel maps it so
// coarser levels (higher index) get heavier weight.
export function strokeForLevel(lvl: number, L: number): number {
  return levelWeight(L - 1 - lvl, L);
}

// ─────────────────────────────────────────────────────────────────────────────
// Glyphs
// ─────────────────────────────────────────────────────────────────────────────
export function GlyphSquare({ color = C.grid, dim = false }: { color?: string; dim?: boolean }) {
  const fill = dim ? 'transparent' : color;
  const stroke = dim ? C.dim : 'rgba(0,0,0,0.3)';
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" style={{ flexShrink: 0, display: 'block' }}>
      <rect x="0.5" y="0.5" width="9" height="9" rx="1" fill={fill} stroke={stroke} strokeWidth="1"/>
    </svg>
  );
}

export function GlyphCircle({ color = C.mesh, dim = false, size = 10 }: { color?: string; dim?: boolean; size?: number }) {
  const fill = dim ? 'transparent' : color;
  const stroke = dim ? C.dim : 'rgba(0,0,0,0.3)';
  const r = size / 2 - 0.5;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ flexShrink: 0, display: 'block' }}>
      <circle cx={size / 2} cy={size / 2} r={r} fill={fill} stroke={stroke} strokeWidth="1"/>
    </svg>
  );
}

export function GlyphArrow({ color, dim = false, weight = 1.5, dash = null }: { color: string; dim?: boolean; weight?: number; dash?: string | null }) {
  const stroke = dim ? C.dim : color;
  return (
    <svg width="16" height="10" viewBox="0 0 16 10" style={{ flexShrink: 0, display: 'block' }}>
      <line x1="1" y1="5" x2="11" y2="5" stroke={stroke} strokeWidth={weight} strokeLinecap="round" strokeDasharray={dash || undefined}/>
      <polyline points="11,2 15,5 11,8" fill="none" stroke={stroke} strokeWidth={weight} strokeLinejoin="round" strokeLinecap="round"/>
    </svg>
  );
}

export function GlyphLine({ color, dim = false, weight = 1.5, dash = null }: { color: string; dim?: boolean; weight?: number; dash?: string | null }) {
  const stroke = dim ? C.dim : color;
  return (
    <svg width="16" height="10" viewBox="0 0 16 10" style={{ flexShrink: 0, display: 'block' }}>
      <line x1="1" y1="5" x2="15" y2="5" stroke={stroke} strokeWidth={weight} strokeLinecap="round" strokeDasharray={dash || undefined}/>
    </svg>
  );
}

export function GlyphArrowVert({ color, dim = false, weight = 1.5, dir = 'up' }: { color: string; dim?: boolean; weight?: number; dir?: 'up' | 'down' }) {
  const stroke = dim ? C.dim : color;
  return (
    <svg width="16" height="10" viewBox="0 0 16 10" style={{ flexShrink: 0, display: 'block' }}>
      {dir === 'up' ? (
        <g>
          <line x1="8" y1="9" x2="8" y2="2" stroke={stroke} strokeWidth={weight} strokeLinecap="round"/>
          <polyline points="5,4 8,1 11,4" fill="none" stroke={stroke} strokeWidth={weight} strokeLinejoin="round" strokeLinecap="round"/>
        </g>
      ) : (
        <g>
          <line x1="8" y1="1" x2="8" y2="8" stroke={stroke} strokeWidth={weight} strokeLinecap="round"/>
          <polyline points="5,6 8,9 11,6" fill="none" stroke={stroke} strokeWidth={weight} strokeLinejoin="round" strokeLinecap="round"/>
        </g>
      )}
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Joint icons for StackHeader
// ─────────────────────────────────────────────────────────────────────────────
function LayerJointIcon({ dim }: { dim: boolean }) {
  const c = dim ? C.dim : C.mesh;
  const op = dim ? 0.45 : 1;
  return (
    <svg width="20" height="10" viewBox="0 0 20 10" style={{ flexShrink: 0, display: 'block', opacity: op }}>
      <circle cx="3" cy="5" r="2.5" fill={c} stroke="rgba(0,0,0,0.3)" strokeWidth="0.8"/>
      <line x1="5.5" y1="5" x2="14.5" y2="5" stroke={C.m2m} strokeWidth="1.4" strokeLinecap="round"/>
      <circle cx="17" cy="5" r="2.5" fill={c} stroke="rgba(0,0,0,0.3)" strokeWidth="0.8"/>
    </svg>
  );
}

function BridgeJointIcon({ dim }: { dim: boolean }) {
  const c = dim ? C.dim : C.m2m;
  const op = dim ? 0.45 : 1;
  return (
    <svg width="20" height="10" viewBox="0 0 20 10" style={{ flexShrink: 0, display: 'block', opacity: op }}>
      <line x1="6" y1="9" x2="6" y2="2" stroke={c} strokeWidth="1.4" strokeLinecap="round"/>
      <polyline points="3.5,4 6,1.5 8.5,4" fill="none" stroke={c} strokeWidth="1.4" strokeLinejoin="round" strokeLinecap="round"/>
      <line x1="14" y1="1" x2="14" y2="8" stroke={c} strokeWidth="1.4" strokeLinecap="round"/>
      <polyline points="11.5,6 14,8.5 16.5,6" fill="none" stroke={c} strokeWidth="1.4" strokeLinejoin="round" strokeLinecap="round"/>
    </svg>
  );
}

function EyeIcon({ on }: { on: boolean }) {
  const col = on ? C.muted : C.dim;
  const o = on ? 0.85 : 0.45;
  return on ? (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style={{ opacity: o, color: col }}>
      <ellipse cx="7" cy="7" rx="5.5" ry="3.5" stroke="currentColor" strokeWidth="1.2"/>
      <circle cx="7" cy="7" r="1.6" fill="currentColor"/>
    </svg>
  ) : (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style={{ opacity: o, color: col }}>
      <ellipse cx="7" cy="7" rx="5.5" ry="3.5" stroke="currentColor" strokeWidth="1.2"/>
      <line x1="2" y1="2" x2="12" y2="12" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
    </svg>
  );
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" style={{
      transition: 'transform 150ms',
      transform: open ? 'rotate(0deg)' : 'rotate(-90deg)',
      flexShrink: 0,
    }}>
      <polyline points="2,3.5 5,6.5 8,3.5" fill="none" stroke={C.dim} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ToggleRow
// ─────────────────────────────────────────────────────────────────────────────
interface ToggleRowProps {
  icon: React.ReactNode;
  label: string;
  level?: string | null;
  meta?: string | null;
  on: boolean;
  indent?: number;
  mode?: 'check' | 'radio';
  onClick: () => void;
  emphasized?: boolean;
  size?: 'md' | 'sm';
  master?: 'on' | 'off' | 'mixed' | null;
  hideEye?: boolean;
  activeRadio?: boolean;
  disabled?: boolean;
}

function ToggleRow({
  icon, label, level = null, meta = null, on, indent = 0, onClick,
  emphasized = false, size = 'md', master = null, hideEye = false, activeRadio = false,
  disabled = false,
}: ToggleRowProps) {
  const stateKey = master ?? (on ? 'on' : 'off');
  const overall = stateKey === 'on' ? 1 : stateKey === 'mixed' ? 0.55 : 0.32;
  const iconOp = stateKey === 'on' ? 1 : stateKey === 'mixed' ? 0.5 : 0.28;
  const labelColor = stateKey === 'on' ? C.text : C.muted;
  const rowH = size === 'sm' ? 24 : 28;
  return (
    <button
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      style={{
        width: '100%', display: 'flex', alignItems: 'center', gap: 8,
        padding: '0 12px 0 ' + (12 + indent * 12 - (activeRadio ? 2 : 0)) + 'px',
        height: rowH,
        background: 'transparent', border: 'none',
        cursor: disabled ? 'not-allowed' : 'pointer',
        fontFamily: SANS, fontSize: emphasized ? 13 : 12.5, color: labelColor,
        textAlign: 'left', position: 'relative',
        transition: 'opacity 120ms, color 120ms',
        borderLeft: activeRadio ? `2px solid ${C.accent}` : '2px solid transparent',
        opacity: disabled ? 0.32 : 1,
      }}
    >
      <span style={{ display: 'inline-flex', width: 16, alignItems: 'center', opacity: iconOp, flexShrink: 0 }}>{icon}</span>
      <span style={{ fontWeight: emphasized ? 600 : 500, opacity: overall, whiteSpace: 'nowrap' }}>{label}</span>
      {level && <span style={{ fontFamily: MONO, fontSize: 10.5, fontWeight: 600, color: C.muted, opacity: overall, marginLeft: 1 }}>{level}</span>}
      {meta && <span style={{ fontSize: 10.5, color: C.dim, marginRight: 'auto', opacity: overall, fontStyle: 'italic' }}>· {meta}</span>}
      {!meta && <span style={{ marginRight: 'auto' }}/>}
      {!hideEye && (
        <span style={{ display: 'inline-flex', alignItems: 'center', flexShrink: 0 }}>
          {stateKey === 'mixed' ? (
            <span style={{ position: 'relative', display: 'inline-flex' }}>
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style={{ color: C.muted, opacity: 0.7 }}>
                <ellipse cx="7" cy="7" rx="5.5" ry="3.5" stroke="currentColor" strokeWidth="1.2"/>
                <circle cx="7" cy="7" r="1.6" fill="currentColor"/>
              </svg>
              <span style={{ position: 'absolute', right: -3, bottom: -2, width: 6, height: 6, borderRadius: '50%', background: C.muted, border: '1.5px solid #fff' }}/>
            </span>
          ) : <EyeIcon on={stateKey === 'on'}/>}
        </span>
      )}
      {activeRadio && <span style={{ marginLeft: 6, width: 6, height: 6, borderRadius: '50%', background: C.accent, flexShrink: 0 }}/>}
    </button>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CollapsibleSection
// ─────────────────────────────────────────────────────────────────────────────
interface CollapsibleSectionProps {
  label: string;
  right?: React.ReactNode;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
  divider?: boolean;
}

function CollapsibleSection({ label, right, open, onToggle, children, divider = true }: CollapsibleSectionProps) {
  return (
    <div>
      {divider && <div style={{ borderTop: `1px solid ${C.border}` }}/>}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 14px 4px' }}>
        <button
          onClick={onToggle}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: 5, padding: 0,
            border: 'none', background: 'transparent', cursor: 'pointer',
            fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase',
            color: C.dim, fontFamily: SANS,
          }}
        >
          <Chevron open={open}/>
          {label}
        </button>
        {right}
      </div>
      <div style={{
        maxHeight: open ? 800 : 0, overflow: 'hidden',
        transition: 'max-height 220ms ease-in-out, opacity 220ms',
        opacity: open ? 1 : 0,
        paddingBottom: open ? 4 : 0,
      }}>
        {children}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// StackHeader
// ─────────────────────────────────────────────────────────────────────────────
interface StackHeaderProps {
  kind: 'layer' | 'bridge';
  label: string;
  sub?: string | null;
  color?: string;
  master?: 'on' | 'off' | 'mixed' | null;
  onMasterToggle?: (() => void) | null;
}

function StackHeader({ kind, label, sub, color = C.dim, master = null, onMasterToggle = null }: StackHeaderProps) {
  const dim = master === 'off';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '6px 14px 2px' }}>
      {kind === 'layer' ? <LayerJointIcon dim={dim}/> : <BridgeJointIcon dim={dim}/>}
      <span style={{ fontSize: 9.5, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase', color, fontFamily: SANS }}>{label}</span>
      {sub && <span style={{ fontSize: 10, color: C.dim, fontStyle: 'italic' }}>· {sub}</span>}
      <span style={{ marginLeft: 'auto' }}/>
      {master && onMasterToggle && (
        <button
          onClick={onMasterToggle}
          title={`Toggle all in ${label}`}
          style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: 0, display: 'inline-flex', alignItems: 'center' }}
        >
          {master === 'mixed' ? (
            <span style={{ position: 'relative', display: 'inline-flex' }}>
              <svg width="13" height="13" viewBox="0 0 14 14" fill="none" style={{ color: C.muted, opacity: 0.7 }}>
                <ellipse cx="7" cy="7" rx="5.5" ry="3.5" stroke="currentColor" strokeWidth="1.2"/>
                <circle cx="7" cy="7" r="1.6" fill="currentColor"/>
              </svg>
              <span style={{ position: 'absolute', right: -3, bottom: -2, width: 5.5, height: 5.5, borderRadius: '50%', background: C.muted, border: '1.4px solid #fff' }}/>
            </span>
          ) : <EyeIcon on={master === 'on'}/>}
        </button>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PanelShell — ported for completeness but never instantiated in integration
// ─────────────────────────────────────────────────────────────────────────────
interface PanelShellProps {
  children: React.ReactNode;
  width?: number;
  mode?: string;
  view?: string;
  onView?: (v: string) => void;
}

export function PanelShell({ children, width = 320, mode = 'Explore', view, onView }: PanelShellProps) {
  return (
    <div style={{
      width, background: C.bgPanel, color: C.text,
      backdropFilter: 'blur(24px)', WebkitBackdropFilter: 'blur(24px)',
      border: `1px solid ${C.border}`, borderRadius: 14,
      boxShadow: '0 4px 28px rgba(0,0,0,.12)',
      display: 'flex', flexDirection: 'column',
      fontFamily: SANS, overflow: 'hidden',
    }}>
      <div style={{ padding: '10px 14px 8px', borderBottom: `1px solid ${C.border}` }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: '-0.01em' }}>
            Logic<span style={{ color: C.accent }}>Lemur</span>
          </div>
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', color: C.dim }}>{mode}</div>
        </div>
      </div>
      {view && onView && (
        <div style={{
          display: 'flex', padding: '6px 8px', gap: 2, borderBottom: `1px solid ${C.border}`,
          background: C.bgRail,
        }}>
          {['diagram', 'list'].map((v) => (
            <button key={v} onClick={() => onView(v)} style={{
              flex: 1, padding: '5px 8px',
              background: view === v ? '#fff' : 'transparent',
              border: view === v ? `1px solid ${C.border}` : '1px solid transparent',
              borderRadius: 6, cursor: 'pointer',
              fontFamily: SANS, fontSize: 11, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase',
              color: view === v ? C.text : C.muted,
              boxShadow: view === v ? '0 1px 2px rgba(0,0,0,.04)' : 'none',
            }}>{v === 'diagram' ? 'Diagram' : 'List'}</button>
          ))}
        </div>
      )}
      {children}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ArchDiagram
// ─────────────────────────────────────────────────────────────────────────────
interface ArchDiagramProps {
  L: number;
  state: Record<string, boolean>;
  toggle: (id: string) => void;
  toggleLayer: (lvl: number) => void;
  hover: string | null;
  setHover: (id: string | null) => void;
  mode?: 'check' | 'radio';
  maxBodyHeight?: number;
  dimmedIds?: Set<string>;
  nodeCounts?: Record<string, number>;
  pinnedId?: string | null;
  onToggleAll?: () => void;
  procMasterState?: 'on' | 'off' | 'mixed';
}

function ArchDiagram({
  L, state, toggle, toggleLayer, hover, setHover,
  mode = 'check', maxBodyHeight = 540, dimmedIds = new Set(), nodeCounts, pinnedId = null,
  onToggleAll, procMasterState,
}: ArchDiagramProps) {
  const svgW = 224;
  const padX = 10;
  const lineW = svgW - padX * 2;  // = 204

  const meshH = L <= 4 ? 38 : L <= 6 ? 30 : 24;
  const bridgeH = L <= 4 ? 32 : L <= 6 ? 24 : 18;

  const nodeCount = (lvl: number) => nodeCounts?.[`mesh_${lvl}`] ?? nodeCounts?.['mesh'] ?? 0;
  const fmtCount = (n: number) => n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k' : String(n);

  const yProcLbl = 2;
  const yMeshTop = 26;
  const totalMeshH = L * meshH + (L - 1) * bridgeH;
  const yMeshBot = yMeshTop + totalMeshH;
  const yEnc = yMeshBot + 4;
  const encH = 46;
  const yGrid = yEnc + encH + 4;
  const gridH = 30;
  const yIO = yGrid + gridH + 4;
  const totalH = yIO + 14;

  const yOfLevel = (lvl: number) => yMeshTop + (L - 1 - lvl) * (meshH + bridgeH);

  const isHov = (id: string) => hover === id;
  const isHighlit = (id: string) => hover === id || pinnedId === id;
  const op = (_id: string, on: boolean) => on ? 1 : 0.32;

  // Derive which items are active from pinnedId
  const pinnedMesh = new Set<number>();
  let pinnedBridgeFiner: number | null = null;
  let pinnedG2M = false;
  let pinnedM2G = false;
  if (pinnedId) {
    if (pinnedId === 'g2m') { pinnedG2M = true; pinnedMesh.add(0); }
    else if (pinnedId === 'm2g') { pinnedM2G = true; pinnedMesh.add(0); }
    else if (pinnedId.startsWith('m2m-')) { pinnedMesh.add(parseInt(pinnedId.slice(4), 10)); }
    else if (pinnedId.startsWith('up-') || pinnedId.startsWith('dn-')) {
      const n = parseInt(pinnedId.slice(3), 10);
      pinnedMesh.add(n); pinnedMesh.add(n + 1);
      pinnedBridgeFiner = n;
    }
  }
  const hasPinned = !!pinnedId;
  // For a given mesh level, is the m2m edge itself the selected subgraph?
  const m2mSelected = (lvl: number) => pinnedId === `m2m-${lvl}`;

  // Encoder/Decoder H-path: connects L0 rail + G2M/M2G box + grid when one is pinned
  let encDecHPath: string | null = null;
  if (pinnedG2M || pinnedM2G) {
    const xCenter = pinnedG2M ? svgW * 0.33 : svgW * 0.67;
    const r = 4;
    const x1 = padX - 2, x2 = padX + lineW + 2;
    const yTop = yOfLevel(0);
    const yMid0 = yTop + meshH;
    const yMid1 = yGrid - 2;
    const yBot = yGrid + gridH + 2;
    const bLeft = xCenter - 30;
    const bRight = xCenter + 30;
    encDecHPath = [
      `M ${x1} ${yTop + r}`, `A ${r},${r} 0 0,1 ${x1 + r},${yTop}`,
      `L ${x2 - r},${yTop}`, `A ${r},${r} 0 0,1 ${x2},${yTop + r}`,
      `L ${x2},${yMid0}`, `L ${bRight},${yMid0}`, `L ${bRight},${yMid1}`, `L ${x2},${yMid1}`,
      `L ${x2},${yBot - r}`, `A ${r},${r} 0 0,1 ${x2 - r},${yBot}`,
      `L ${x1 + r},${yBot}`, `A ${r},${r} 0 0,1 ${x1},${yBot - r}`,
      `L ${x1},${yMid1}`, `L ${bLeft},${yMid1}`, `L ${bLeft},${yMid0}`, `L ${x1},${yMid0}`, `Z`,
    ].join(' ');
  }

  const meshRail = (lvl: number) => {
    const y = yOfLevel(lvl);
    const meshOn = state[`mesh-${lvl}`];
    const m2mOn = state[`m2m-${lvl}`];
    const layerId = `layer-${lvl}`;
    const isLevelActive = !hasPinned || pinnedMesh.has(lvl);
    const layerHover = isHov(`mesh-${lvl}`) || isHov(`m2m-${lvl}`) || isHov(layerId);
    // When a bridge is pinned, the H-path covers both connected rails — suppress individual rail box
    const bridgeOwnsHighlight = pinnedBridgeFiner !== null &&
      (lvl === pinnedBridgeFiner || lvl === pinnedBridgeFiner + 1);
    // When g2m/m2g is pinned, the enc-dec H-path covers L0 rail — suppress its individual box
    const encDecOwnsHighlight = (pinnedG2M || pinnedM2G) && lvl === 0;
    const showHighlight = !bridgeOwnsHighlight && !encDecOwnsHighlight && (layerHover || (hasPinned && pinnedMesh.has(lvl)));
    const n = Math.min(20, 3 + (L - 1 - lvl) * 4);
    const dotR = 2.2 + 1.0 * lvl / Math.max(1, L - 1);
    // m2m line: dim when a bridge/g2m/m2g is selected (only nodes should stand out)
    const m2mLineOp = (!hasPinned || m2mSelected(lvl)) ? (m2mOn ? 1 : 0.32) : 0.18;

    return (
      <g key={`rail-${lvl}`}
         onMouseEnter={() => setHover(layerId)} onMouseLeave={() => setHover(null)}
         onClick={() => toggleLayer(lvl)}
         style={{ cursor: 'pointer', opacity: isLevelActive ? 1 : 0.25 }}>
        <rect x={padX - 2} y={y} width={lineW + 4} height={meshH} rx={4}
          fill={showHighlight ? 'rgba(0,172,193,0.05)' : 'transparent'}
          stroke={showHighlight ? C.accent : C.borderL} strokeWidth={1}/>
        <line x1={padX + 4} y1={y + meshH / 2} x2={padX + lineW - 4} y2={y + meshH / 2}
          stroke={C.m2m} strokeWidth={strokeForLevel(lvl, L)}
          strokeLinecap="round" opacity={m2mLineOp}/>
        {Array.from({ length: n }).map((_, k) => {
          const x = padX + ((k + 0.5) / n) * lineW;
          return (
            <circle key={k} cx={x} cy={y + meshH / 2} r={dotR}
              fill={meshOn ? C.mesh : 'transparent'}
              stroke={meshOn ? 'rgba(0,0,0,0.4)' : C.dim} strokeWidth="0.7"
              opacity={meshOn ? 1 : 0.45}/>
          );
        })}
      </g>
    );
  };

  const bridge = (upperLevel: number, lowerLevel: number) => {
    const yUp = yOfLevel(upperLevel);
    const yLow = yOfLevel(lowerLevel);
    const yMid0 = yUp + meshH;
    const yMid1 = yLow;
    const upId = `up-${lowerLevel}`;
    const dnId = `dn-${lowerLevel}`;
    const cx = padX + lineW / 2;
    const upOn = state[upId], dnOn = state[dnId];
    const wWeight = strokeForLevel(upperLevel, L);
    const midY = (yMid0 + yMid1) / 2;

    const isPinnedUp = hasPinned && pinnedId === upId;
    const isPinnedDn = hasPinned && pinnedId === dnId;
    const isBridgePinned = isPinnedUp || isPinnedDn;
    const isBridgeActive = !hasPinned || isBridgePinned;
    const bH = yMid1 - yMid0 - 2;

    // H-shape path: two full-width rail boxes joined by the selected arrow's narrow connector
    let hShapePath: string | null = null;
    if (isBridgePinned) {
      const r = 4;
      const x1 = padX - 2, x2 = padX + lineW + 2;
      const yTop = yOfLevel(upperLevel), yBot = yOfLevel(lowerLevel) + meshH;
      const bLeft  = isPinnedUp ? cx - 54 : cx + 4;
      const bRight = isPinnedUp ? cx - 4  : cx + 54;
      hShapePath = [
        `M ${x1} ${yTop + r}`,
        `A ${r},${r} 0 0,1 ${x1 + r},${yTop}`,
        `L ${x2 - r},${yTop}`,
        `A ${r},${r} 0 0,1 ${x2},${yTop + r}`,
        `L ${x2},${yMid0}`,
        `L ${bRight},${yMid0}`,
        `L ${bRight},${yMid1}`,
        `L ${x2},${yMid1}`,
        `L ${x2},${yBot - r}`,
        `A ${r},${r} 0 0,1 ${x2 - r},${yBot}`,
        `L ${x1 + r},${yBot}`,
        `A ${r},${r} 0 0,1 ${x1},${yBot - r}`,
        `L ${x1},${yMid1}`,
        `L ${bLeft},${yMid1}`,
        `L ${bLeft},${yMid0}`,
        `L ${x1},${yMid0}`,
        `Z`,
      ].join(' ');
    }

    return (
      <g key={`bridge-${lowerLevel}`} style={{ opacity: isBridgeActive ? 1 : 0.25 }}>
        {hShapePath && (
          <path d={hShapePath} fill="rgba(0,172,193,0.05)" stroke={C.accent} strokeWidth="1"
            style={{ pointerEvents: 'none' }}/>
        )}
        {/* MeshUp: finer→coarser */}
        <g onMouseEnter={() => setHover(upId)} onMouseLeave={() => setHover(null)}
           onClick={() => toggle(upId)} style={{ cursor: 'pointer' }}>
          {!isBridgePinned && (
            <rect x={cx - 54} y={yMid0 + 1} width={50} height={bH} rx={3}
              fill={isHov(upId) ? 'rgba(0,172,193,0.05)' : 'transparent'}
              stroke={isHov(upId) ? C.accent : 'transparent'} strokeWidth="1"/>
          )}
          <line x1={cx - 22} y1={yMid1} x2={cx - 22} y2={yMid0 + 2}
            stroke={C.m2m} strokeWidth={wWeight} strokeLinecap="round" opacity={op(upId, upOn)}/>
          <polyline points={`${cx - 25},${yMid0 + 5} ${cx - 22},${yMid0 + 2} ${cx - 19},${yMid0 + 5}`}
            fill="none" stroke={C.m2m} strokeWidth={wWeight} strokeLinejoin="round" strokeLinecap="round"
            opacity={op(upId, upOn)}/>
          <text x={cx - 28} y={midY + 3} fontSize="8" fontFamily={MONO}
            textAnchor="end" fill={C.muted} opacity={op(upId, upOn) * 0.8 + 0.2}>{lowerLevel}→{upperLevel}</text>
          {isHov(upId) && (
            <text x={padX} y={midY + 3} fontSize="8" fontFamily={SANS} fontWeight="600"
              fill={C.text} textAnchor="start">MeshUp</text>
          )}
        </g>
        {/* MeshDown: coarser→finer */}
        <g onMouseEnter={() => setHover(dnId)} onMouseLeave={() => setHover(null)}
           onClick={() => toggle(dnId)} style={{ cursor: 'pointer' }}>
          {!isBridgePinned && (
            <rect x={cx + 4} y={yMid0 + 1} width={50} height={bH} rx={3}
              fill={isHov(dnId) ? 'rgba(0,172,193,0.05)' : 'transparent'}
              stroke={isHov(dnId) ? C.accent : 'transparent'} strokeWidth="1"/>
          )}
          <line x1={cx + 22} y1={yMid0} x2={cx + 22} y2={yMid1 - 2}
            stroke={C.m2m} strokeWidth={wWeight} strokeLinecap="round" opacity={op(dnId, dnOn)}/>
          <polyline points={`${cx + 19},${yMid1 - 5} ${cx + 22},${yMid1 - 2} ${cx + 25},${yMid1 - 5}`}
            fill="none" stroke={C.m2m} strokeWidth={wWeight} strokeLinejoin="round" strokeLinecap="round"
            opacity={op(dnId, dnOn)}/>
          <text x={cx + 28} y={midY + 3} fontSize="8" fontFamily={MONO}
            fill={C.muted} opacity={op(dnId, dnOn) * 0.8 + 0.2}>{upperLevel}→{lowerLevel}</text>
          {isHov(dnId) && (
            <text x={padX + lineW} y={midY + 3} fontSize="8" fontFamily={SANS} fontWeight="600"
              fill={C.text} textAnchor="end">MeshDown</text>
          )}
        </g>
      </g>
    );
  };

  const gridSquares = 20;
  const gridSqW = 8;
  const gridSqGap = (lineW - gridSquares * gridSqW) / (gridSquares - 1);

  // Label column height sections (must sum to totalH):
  // paddingTop=yMeshTop, then level blocks, then enc section, grid section, IO
  const encSectionH = 4 + encH;         // gap after mesh + encH
  const gridSectionH = 4 + gridH;       // gap after enc + gridH
  const ioSectionH = 4 + 14;            // gap after grid + IO label height

  return (
    <div style={{ position: 'relative' }}>
      {onToggleAll && mode === 'check' && (
        <button
          onClick={onToggleAll}
          style={{
            position: 'absolute', top: 9, right: 98, zIndex: 1,
            fontSize: 8.5, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase',
            color: procMasterState === 'on' ? C.accent : C.dim,
            background: 'transparent', border: 'none', cursor: 'pointer',
            fontFamily: SANS, padding: 0, lineHeight: 1,
          }}
        >
          {procMasterState === 'on' ? 'hide all' : 'show all'}
        </button>
      )}
    <div style={{ display: 'flex', padding: '6px 2px', maxHeight: maxBodyHeight, overflowY: totalH > maxBodyHeight ? 'auto' : 'visible' }}>

      {/* Icon-only SVG */}
      <svg width={svgW} height={totalH} style={{ display: 'block', overflow: 'visible', flexShrink: 0 }}>
        <rect x={padX - 2} y={yMeshTop - 6} width={lineW + 4} height={totalMeshH + 12}
          rx={6} fill={C.bgTint} stroke={C.borderL} strokeWidth="0.8"/>
        <text x={padX} y={yProcLbl + 8} fontSize="8.5" fontWeight="700" letterSpacing="1.5" fill={C.dim} fontFamily={SANS}>PROCESSOR</text>

        <g id="toggle-m2m">
          <g id="toggle-mesh">{Array.from({ length: L }).map((_, lvl) => meshRail(lvl))}</g>
          {Array.from({ length: L - 1 }).map((_, i) => bridge(i + 1, i))}
        </g>

        {encDecHPath && (
          <path d={encDecHPath} fill="rgba(0,172,193,0.05)" stroke={C.accent} strokeWidth="1"
            style={{ pointerEvents: 'none' }}/>
        )}

        {/* G2M and M2G — keep color-coded labels */}
        {(() => {
          const xG = svgW * 0.33, xM = svgW * 0.67;
          const yA0 = yEnc + 4, yA1 = yEnc + encH - 4;
          const g2m = state.g2m, m2g = state.m2g;
          return (
            <g>
              <g id="toggle-g2m" onMouseEnter={() => setHover('g2m')} onMouseLeave={() => setHover(null)}
                 onClick={() => toggle('g2m')}
                 style={{ cursor: 'pointer', opacity: hasPinned && !pinnedG2M ? 0.25 : 1 }}>
                <rect x={xG - 30} y={yEnc} width={60} height={encH} rx={4}
                  fill={isHighlit('g2m') ? 'rgba(0,172,193,0.05)' : 'transparent'}
                  stroke={isHighlit('g2m') && !pinnedG2M ? C.accent : 'transparent'} strokeWidth="1"/>
                <line x1={xG} y1={yA1} x2={xG} y2={yA0 + 2} stroke={C.g2m} strokeWidth="2"
                  strokeLinecap="round" opacity={op('g2m', g2m)}/>
                <polyline points={`${xG - 3.5},${yA0 + 5} ${xG},${yA0 + 2} ${xG + 3.5},${yA0 + 5}`}
                  fill="none" stroke={C.g2m} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
                  opacity={op('g2m', g2m)}/>
                <text x={xG} y={yEnc + encH / 2 + 4} textAnchor="middle" fontSize="9" fontWeight="600"
                  fontFamily={MONO} fill={g2m ? C.g2m : C.dim} opacity={op('g2m', g2m)} dx="-14">G2M</text>
              </g>
              <g id="toggle-m2g" onMouseEnter={() => setHover('m2g')} onMouseLeave={() => setHover(null)}
                 onClick={() => toggle('m2g')}
                 style={{ cursor: 'pointer', opacity: hasPinned && !pinnedM2G ? 0.25 : 1 }}>
                <rect x={xM - 30} y={yEnc} width={60} height={encH} rx={4}
                  fill={isHighlit('m2g') ? 'rgba(0,172,193,0.05)' : 'transparent'}
                  stroke={isHighlit('m2g') && !pinnedM2G ? C.accent : 'transparent'} strokeWidth="1"/>
                <line x1={xM} y1={yA0} x2={xM} y2={yA1 - 2} stroke={C.m2g} strokeWidth="2"
                  strokeLinecap="round" opacity={op('m2g', m2g)}/>
                <polyline points={`${xM - 3.5},${yA1 - 5} ${xM},${yA1 - 2} ${xM + 3.5},${yA1 - 5}`}
                  fill="none" stroke={C.m2g} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
                  opacity={op('m2g', m2g)}/>
                <text x={xM} y={yEnc + encH / 2 + 4} textAnchor="middle" fontSize="9" fontWeight="600"
                  fontFamily={MONO} fill={m2g ? C.m2g : C.dim} opacity={op('m2g', m2g)} dx="14">M2G</text>
              </g>
            </g>
          );
        })()}

        {/* Grid squares — no count label */}
        <g id="toggle-grid" onMouseEnter={() => setHover('grid')} onMouseLeave={() => setHover(null)}
           onClick={(dimmedIds.has('grid') || mode === 'radio') ? undefined : () => toggle('grid')}
           style={{ cursor: (dimmedIds.has('grid') || mode === 'radio') ? 'default' : 'pointer', opacity: dimmedIds.has('grid') ? 0.32 : 1 }}>
          <rect x={padX - 2} y={yGrid - 2} width={lineW + 4} height={gridH + 4} rx={4} fill="transparent"
            stroke={isHov('grid') && !(pinnedG2M || pinnedM2G) ? C.accent : 'transparent'} strokeWidth="1"/>
          {Array.from({ length: gridSquares }).map((_, i) => {
            const x = padX + i * (gridSqW + gridSqGap);
            return (
              <rect key={i} x={x} y={yGrid + 6} width={gridSqW} height={gridSqW} rx="1.5"
                fill={state.grid ? C.grid : 'transparent'}
                stroke={state.grid ? 'rgba(0,0,0,0.3)' : C.dim} strokeWidth="0.7"
                opacity={state.grid ? 1 : 0.5}/>
            );
          })}
        </g>

        <text x={padX} y={yIO + 8} fontSize="8" fontFamily={SANS} fontWeight="700" fill={C.dim} letterSpacing="1.2">INPUT</text>
        <text x={svgW - padX} y={yIO + 8} fontSize="8" fontFamily={SANS} fontWeight="700" fill={C.dim} letterSpacing="1.2" textAnchor="end">OUTPUT</text>
      </svg>

      {/* Label column — vertically aligned to SVG sections */}
      <div style={{ width: 88, boxSizing: 'border-box', paddingTop: yMeshTop, display: 'flex', flexDirection: 'column' }}>
        {Array.from({ length: L }, (_, i) => L - 1 - i).map((lvl) => {
          const cnt = nodeCount(lvl);
          const active = state[`mesh-${lvl}`] || state[`m2m-${lvl}`];
          return (
            <React.Fragment key={lvl}>
              {/* Level row */}
              <div onClick={() => toggleLayer(lvl)}
                style={{ height: meshH, boxSizing: 'border-box', cursor: 'pointer', paddingTop: 4, paddingLeft: 6, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 1 }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
                  <span style={{ fontFamily: MONO, fontSize: 10, fontWeight: 700, color: active ? C.mesh : C.dim, letterSpacing: '-0.01em', lineHeight: 1 }}>{L <= 1 ? 'Mesh' : `L${lvl}`}</span>
                  {cnt > 0 && <span style={{ fontFamily: MONO, fontSize: 8, color: C.muted, lineHeight: 1 }}>{fmtCount(cnt)} nodes</span>}
                </div>
                {L > 1 && (lvl === 0 || lvl === L - 1) && (
                  <span style={{ fontFamily: SANS, fontSize: 7.5, color: C.dim, fontStyle: 'italic', lineHeight: 1 }}>
                    {lvl === 0 ? 'finest mesh' : 'coarsest mesh'}
                  </span>
                )}
              </div>
              {/* Bridge row (between this level and the finer one below it) */}
              {lvl > 0 && (
                <div style={{ height: bridgeH, boxSizing: 'border-box', paddingLeft: 6, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 2 }}>
                  <span style={{ fontFamily: MONO, fontSize: 7.5, lineHeight: 1 }}>
                    <span style={{ color: state[`up-${lvl - 1}`] ? C.m2m : C.dim }}>L{lvl - 1}→L{lvl}</span>
                    <span style={{ color: C.dim }}>{' '}MeshUp</span>
                  </span>
                  <span style={{ fontFamily: MONO, fontSize: 7.5, lineHeight: 1 }}>
                    <span style={{ color: state[`dn-${lvl - 1}`] ? C.m2m : C.dim }}>L{lvl}→L{lvl - 1}</span>
                    <span style={{ color: C.dim }}>{' '}MeshDown</span>
                  </span>
                </div>
              )}
            </React.Fragment>
          );
        })}
        {/* Encoder / decoder label — aligned to G2M+M2G SVG section */}
        <div style={{ height: encSectionH, boxSizing: 'border-box', paddingLeft: 6, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 3 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
            <span style={{ fontFamily: MONO, fontSize: 9, fontWeight: 700, color: state.g2m ? C.g2m : C.dim, lineHeight: 1 }}>G2M</span>
            <span style={{ fontFamily: SANS, fontSize: 7.5, color: C.dim, lineHeight: 1 }}>encoder</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
            <span style={{ fontFamily: MONO, fontSize: 9, fontWeight: 700, color: state.m2g ? C.m2g : C.dim, lineHeight: 1 }}>M2G</span>
            <span style={{ fontFamily: SANS, fontSize: 7.5, color: C.dim, lineHeight: 1 }}>decoder</span>
          </div>
        </div>
        {/* Grid label — aligned to grid squares SVG section */}
        <div style={{ height: gridSectionH, boxSizing: 'border-box', paddingLeft: 6, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 1 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
            <span style={{ fontFamily: MONO, fontSize: 10, fontWeight: 700, color: state.grid ? C.grid : C.dim, lineHeight: 1 }}>Grid</span>
            {(nodeCounts?.['grid'] ?? 0) > 0 && (
              <span style={{ fontFamily: MONO, fontSize: 8, color: C.muted, lineHeight: 1 }}>
                {fmtCount(nodeCounts!['grid'])} nodes
              </span>
            )}
          </div>
          <span style={{ fontFamily: SANS, fontSize: 7.5, color: C.dim, fontStyle: 'italic', lineHeight: 1 }}>lat/lon pts</span>
        </div>
        {/* IO spacer */}
        <div style={{ height: ioSectionH }}/>
      </div>
    </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// StackedSidebar — main exported component
// ─────────────────────────────────────────────────────────────────────────────
export interface StackedSidebarProps {
  L?: number;
  mode?: 'check' | 'radio';
  nodeCounts?: Record<string, number>;
  onStateChange?: (activeSubgraphs: Set<string>, activeNodeTypes: Set<string>) => void;
  value?: string;  // controlled: API subgraph key — overrides internal radio selection when provided
}

export function StackedSidebar({ L = 1, mode = 'check', nodeCounts, onStateChange, value }: StackedSidebarProps) {
  const ids = useMemo(() => ({
    mesh: Array.from({ length: L }, (_, i) => `mesh-${i}`),
    m2m: Array.from({ length: L }, (_, i) => `m2m-${i}`),
    up: Array.from({ length: Math.max(0, L - 1) }, (_, i) => `up-${i}`),
    dn: Array.from({ length: Math.max(0, L - 1) }, (_, i) => `dn-${i}`),
  }), [L]);

  const allEdgeIds = useMemo(() => [...ids.m2m, ...ids.up, ...ids.dn, 'g2m', 'm2g'], [ids]);

  const defaults = useMemo(() => {
    const o: Record<string, boolean> = { grid: true, g2m: true, m2g: true };
    ids.mesh.forEach((id) => { o[id] = true; });
    ids.m2m.forEach((id) => { o[id] = mode === 'radio' ? false : true; });
    ids.up.forEach((id) => { o[id] = mode === 'radio' ? false : true; });
    ids.dn.forEach((id) => { o[id] = false; });
    if (mode === 'radio') {
      o.g2m = false; o.m2g = false;
      o[`m2m-${Math.floor(L / 2)}`] = true;
    }
    return o;
  }, [L, mode, ids]);

  const [s, setS] = useState<Record<string, boolean>>(defaults);
  const [hover, setHover] = useState<string | null>(null);

  // Translate external API subgraph key to internal UI key
  const valueUiKey = useMemo((): string | null => {
    if (!value || mode !== 'radio') return null;
    if (value === 'g2m') return 'g2m';
    if (value === 'm2g') return 'm2g';
    if (value === 'm2m') return 'm2m-0';  // single-scale
    const m2m = value.match(/^m2m_(\d+)$/); if (m2m) return `m2m-${m2m[1]}`;
    const up  = value.match(/^mesh_up_(\d+)$/); if (up) return `up-${up[1]}`;
    const dn  = value.match(/^mesh_down_(\d+)$/); if (dn) return `dn-${dn[1]}`;
    return null;
  }, [value, mode]);

  // In radio mode: which mesh levels are actually connected by the selected subgraph
  const activeMeshLevels = useMemo((): Set<number> => {
    if (mode !== 'radio' || !valueUiKey) return new Set(Array.from({ length: L }, (_, i) => i));
    if (valueUiKey === 'g2m' || valueUiKey === 'm2g') return new Set([0]);
    const m2m = valueUiKey.match(/^m2m-(\d+)$/); if (m2m) return new Set([parseInt(m2m[1], 10)]);
    const bridge = valueUiKey.match(/^(?:up|dn)-(\d+)$/);
    if (bridge) { const n = parseInt(bridge[1], 10); return new Set([n, n + 1]); }
    return new Set();
  }, [mode, valueUiKey, L]);

  // Default view from localStorage
  const [view, setView] = useState<'diagram' | 'list'>(() =>
    typeof window !== 'undefined'
      ? ((localStorage.getItem('panel-view') as 'diagram' | 'list') ?? 'diagram')
      : 'diagram',
  );

  // Collapse state from localStorage
  const readCollapse = (k: string) => {
    if (typeof window === 'undefined') return true;
    const v = localStorage.getItem('panel-collapse-' + k);
    return v === null || v !== 'false';
  };

  const [open, setOpen] = useState<Record<string, boolean>>({
    nodes: readCollapse('nodes'),
    encoder: readCollapse('encoder'),
    processor: readCollapse('processor'),
    decoder: readCollapse('decoder'),
    io: readCollapse('io'),
  });

  const handleView = (v: 'diagram' | 'list') => {
    setView(v);
    if (typeof window !== 'undefined') localStorage.setItem('panel-view', v);
  };

  const setOpenKey = (k: string, val: boolean) => {
    setOpen((o) => {
      const next = { ...o, [k]: val };
      if (typeof window !== 'undefined') localStorage.setItem('panel-collapse-' + k, String(val));
      return next;
    });
  };

  // Keyboard shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === 'g') handleView('diagram');
      if (e.key === 'l') handleView('list');
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  // Selection state resets when the graph (L) or panel mode changes, and follows the
  // externally controlled `value` in radio mode. Because `value` wins over the built-in
  // radio default, re-entering inspect re-opens the last inspected subgraph.
  const prevDefaults = useRef(defaults);
  useEffect(() => {
    const defaultsChanged = prevDefaults.current !== defaults;
    prevDefaults.current = defaults;
    setS((cur) => {
      const base = defaultsChanged ? defaults : cur;
      if (mode !== 'radio' || !valueUiKey || !(valueUiKey in defaults)) return base;
      if (base[valueUiKey]) return base; // already active — no-op prevents loop
      const next = { ...base };
      allEdgeIds.forEach((eid) => { next[eid] = false; });
      next[valueUiKey] = true;
      return next;
    });
  }, [defaults, valueUiKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = (id: string) => {
    if (mode === 'radio' && allEdgeIds.includes(id)) {
      const next = { ...s };
      allEdgeIds.forEach((eid) => { next[eid] = false; });
      next[id] = true;
      setS(next);
    } else {
      setS((o) => ({ ...o, [id]: !o[id] }));
    }
  };

  const masterState = (g: string[]): 'on' | 'off' | 'mixed' => {
    const v = g.map((id) => s[id]);
    return v.every(Boolean) ? 'on' : v.every((x) => !x) ? 'off' : 'mixed';
  };

  const toggleGroup = (g: string[]) => {
    if (mode === 'radio') return;
    const ms = masterState(g);
    const next = ms === 'on' ? false : true;
    setS((o) => {
      const x = { ...o };
      g.forEach((id) => { x[id] = next; });
      return x;
    });
  };

  const toggleLayer = (lvl: number) => {
    if (mode === 'radio') { toggle(`m2m-${lvl}`); return; }
    toggleGroup([`mesh-${lvl}`, `m2m-${lvl}`]);
  };

  // Fix 3: inspectDim — G2M/M2G connect to mesh level 0 (finest)
  const inspectDim = useMemo(() => {
    if (mode !== 'radio') return new Set<string>();
    const active = allEdgeIds.find((id) => s[id]);
    if (!active) return new Set<string>();
    const activeMesh = new Set<number>();
    let gridParticipates = false;
    if (active === 'g2m' || active === 'm2g') {
      activeMesh.add(0); // mesh_0 is always finest (connects to grid), NOT L-1
      gridParticipates = true;
    } else if (active.startsWith('m2m-')) {
      activeMesh.add(parseInt(active.slice(4), 10));
    } else if (active.startsWith('up-') || active.startsWith('dn-')) {
      const k = parseInt(active.slice(3), 10);
      activeMesh.add(k); activeMesh.add(k + 1);
    }
    const dim = new Set<string>();
    for (let k = 0; k < L; k++) if (!activeMesh.has(k)) dim.add(`mesh-${k}`);
    if (!gridParticipates) dim.add('grid');
    return dim;
  }, [mode, s, L, allEdgeIds]);

  const isDim = (id: string) => inspectDim.has(id);

  const procEdges = [...ids.mesh, ...ids.m2m, ...ids.up, ...ids.dn];

  // onStateChange callback
  useEffect(() => {
    if (!onStateChange) return;
    const activeSubs = new Set<string>();
    const activeNodes = new Set<string>();

    // Build active subgraphs (UI keys)
    if (s['g2m']) activeSubs.add('g2m');
    if (s['m2g']) activeSubs.add('m2g');
    ids.m2m.forEach((id, i) => { if (s[id]) activeSubs.add(`m2m-${i}`); });
    ids.up.forEach((id, i) => { if (s[id]) activeSubs.add(`up-${i}`); });
    ids.dn.forEach((id, i) => { if (s[id]) activeSubs.add(`dn-${i}`); });

    // Build active node types (UI keys)
    if (s['grid']) activeNodes.add('grid');
    ids.mesh.forEach((id, i) => { if (s[id]) activeNodes.add(`mesh-${i}`); });

    onStateChange(activeSubs, activeNodes);
  }, [s]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <>
      {/* View tab bar — same style as mock PanelShell view tabs */}
      <div style={{
        display: 'flex', padding: '6px 8px', gap: 2,
        borderBottom: `1px solid ${C.border}`,
        background: C.bgRail,
        flexShrink: 0,
      }}>
        {(['diagram', 'list'] as const).map((v) => (
          <button key={v} onClick={() => handleView(v)} style={{
            flex: 1, padding: '5px 8px',
            background: view === v ? '#fff' : 'transparent',
            border: view === v ? `1px solid ${C.border}` : '1px solid transparent',
            borderRadius: 6, cursor: 'pointer',
            fontFamily: SANS, fontSize: 11, fontWeight: 600,
            letterSpacing: '0.06em', textTransform: 'uppercase',
            color: view === v ? C.text : C.muted,
            boxShadow: view === v ? '0 1px 2px rgba(0,0,0,.04)' : 'none',
          }}>{v === 'diagram' ? 'Diagram' : 'List'}</button>
        ))}
      </div>

      {view === 'diagram' ? (
        <ArchDiagram
          L={L} state={s} toggle={toggle} toggleLayer={toggleLayer}
          hover={hover} setHover={setHover} mode={mode} dimmedIds={inspectDim}
          nodeCounts={nodeCounts} pinnedId={valueUiKey}
          onToggleAll={() => toggleGroup(procEdges)}
          procMasterState={masterState(procEdges)}
        />
      ) : (
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {/* PROCESSOR */}
          <CollapsibleSection
            label={`Processor · ${L} level${L > 1 ? 's' : ''}`}
            open={open.processor}
            onToggle={() => setOpenKey('processor', !open.processor)}
            right={mode === 'check' ? (
              <button
                onClick={(e) => { e.stopPropagation(); toggleGroup(procEdges); }}
                style={{
                  fontSize: 9.5, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase',
                  color: masterState(procEdges) === 'on' ? C.accent : C.dim,
                  background: 'transparent', border: 'none', cursor: 'pointer', fontFamily: SANS, padding: 0,
                }}
              >
                {masterState(procEdges) === 'on' ? 'hide all' : 'show all'}
              </button>
            ) : null}
          >
            {/* Iterate DESCENDING: coarsest first at top, finest last at bottom */}
            {Array.from({ length: L }, (_, i) => L - 1 - i).map((level) => {
              const layerIds = [`mesh-${level}`, `m2m-${level}`];
              const layerMaster = masterState(layerIds);
              const sub = level === L - 1 ? 'coarsest mesh' : level === 0 ? 'finest mesh' : null;
              const lvlTag = L > 1 ? `L${level}` : null;
              return (
                <React.Fragment key={`stack-${level}`}>
                  {L > 1 && (
                    <StackHeader
                      kind="layer" label={`Layer L${level}`}
                      sub={sub}
                      master={mode === 'check' ? layerMaster : null}
                      onMasterToggle={mode === 'check' ? () => toggleGroup(layerIds) : null}
                    />
                  )}
                  <ToggleRow
                    indent={1} size="sm" mode={mode}
                    icon={<GlyphCircle color={C.mesh} dim={mode === 'radio' ? !activeMeshLevels.has(level) : !s[`mesh-${level}`]}/>}
                    label="Mesh nodes" level={lvlTag}
                    on={mode === 'radio' ? activeMeshLevels.has(level) : s[`mesh-${level}`]}
                    onClick={() => toggle(`mesh-${level}`)}
                    hideEye={mode === 'radio'}
                  />
                  <ToggleRow
                    indent={1} size="sm" mode={mode}
                    icon={<GlyphLine color={C.m2m} dim={!s[`m2m-${level}`]} weight={strokeForLevel(level, L)}/>}
                    label="M2M" level={lvlTag}
                    meta={mode === 'radio' ? null : 'intra-level'}
                    on={s[`m2m-${level}`]} onClick={() => toggle(`m2m-${level}`)}
                    activeRadio={mode === 'radio' && s[`m2m-${level}`]}
                  />

                  {/* Bridge: after each level entry, render bridge if level > 0 */}
                  {level > 0 && (() => {
                    // Bridge between level and level-1 (level is upper/coarser, level-1 is lower/finer)
                    const finerLevel = level - 1;
                    const upId = `up-${finerLevel}`;
                    const dnId = `dn-${finerLevel}`;
                    const bridgeIds = [upId, dnId];
                    const bridgeMaster = masterState(bridgeIds);
                    return (
                      <>
                        <StackHeader
                          kind="bridge" label={`Bridge L${level} ↔ L${finerLevel}`}
                          master={mode === 'check' ? bridgeMaster : null}
                          onMasterToggle={mode === 'check' ? () => toggleGroup(bridgeIds) : null}
                        />
                        {/* MeshUp: finer→coarser. Label: L${finerLevel}→L${level} */}
                        <ToggleRow
                          indent={1} size="sm" mode={mode}
                          icon={<GlyphArrow color={C.m2m} dim={!s[upId]} weight={strokeForLevel(finerLevel, L)}/>}
                          label="MeshUp" level={`L${finerLevel}→L${level}`}
                          on={s[upId]} onClick={() => toggle(upId)}
                          activeRadio={mode === 'radio' && s[upId]}
                        />
                        {/* MeshDown: coarser→finer. Label: L${level}→L${finerLevel} */}
                        <ToggleRow
                          indent={1} size="sm" mode={mode}
                          icon={<GlyphArrow color={C.m2m} dim={!s[dnId]} weight={strokeForLevel(finerLevel, L)}/>}
                          label="MeshDown" level={`L${level}→L${finerLevel}`}
                          on={s[dnId]} onClick={() => toggle(dnId)}
                          activeRadio={mode === 'radio' && s[dnId]}
                        />
                      </>
                    );
                  })()}
                </React.Fragment>
              );
            })}
          </CollapsibleSection>

          {/* ENCODER */}
          <CollapsibleSection
            label="Encoder"
            open={open.encoder}
            onToggle={() => setOpenKey('encoder', !open.encoder)}
          >
            <ToggleRow
              mode={mode}
              icon={<GlyphArrow color={C.g2m} dim={!s.g2m}/>}
              label="G2M" meta="grid → mesh L0"
              on={s.g2m} onClick={() => toggle('g2m')}
              activeRadio={mode === 'radio' && s.g2m}
            />
          </CollapsibleSection>

          {/* DECODER */}
          <CollapsibleSection
            label="Decoder"
            open={open.decoder}
            onToggle={() => setOpenKey('decoder', !open.decoder)}
          >
            <ToggleRow
              mode={mode}
              icon={<GlyphArrow color={C.m2g} dim={!s.m2g}/>}
              label="M2G" meta="mesh L0 → grid"
              on={s.m2g} onClick={() => toggle('m2g')}
              activeRadio={mode === 'radio' && s.m2g}
            />
          </CollapsibleSection>

          {/* INPUT / OUTPUT */}
          <CollapsibleSection
            label="Input / Output"
            open={open.io ?? true}
            onToggle={() => setOpenKey('io', !(open.io ?? true))}
            right={
              <HintButton placement="left" content={HINT_GRID_NODES} />
            }
          >
            <ToggleRow
              mode="check"
              icon={<GlyphSquare color={C.grid} dim={!s.grid}/>}
              label="Grid nodes" meta="ERA5 lat/lon points"
              on={s.grid} onClick={() => toggle('grid')} disabled={isDim('grid')}
            />
          </CollapsibleSection>

          <div style={{ height: 10 }}/>
        </div>
      )}
    </>
  );
}

// Suppress unused-variable hints
void levelDash; void GlyphArrowVert;
