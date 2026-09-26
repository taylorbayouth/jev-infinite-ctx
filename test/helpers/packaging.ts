/**
 * Packaging test support: a fresh-checkout copy of the working tree, `npm
 * pack --dry-run` file lists, and a build into a scratch directory. Nothing
 * here reads or writes the repo's own dist/.
 */

import { execFileSync } from "node:child_process";
import { cpSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, statSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

/**
 * Copies the working tree's tracked and untracked-but-not-ignored files into
 * a temp directory, so dist/ is absent exactly as after `git clone`, and
 * links the repo's node_modules so lifecycle scripts can find tsc. The
 * caller removes the directory.
 */
export function freshCheckout(): string {
  const listed = execFileSync("git", ["ls-files", "--cached", "--others", "--exclude-standard", "-z"], {
    cwd: repoRoot,
    encoding: "utf8",
  })
    .split("\0")
    .filter((file) => file.length > 0);
  const dir = mkdtempSync(path.join(tmpdir(), "jev-checkout-"));
  for (const rel of listed) {
    const dest = path.join(dir, rel);
    mkdirSync(path.dirname(dest), { recursive: true });
    try {
      cpSync(path.join(repoRoot, rel), dest);
    } catch {
      // A tracked file deleted in the working tree.
    }
  }
  symlinkSync(path.join(repoRoot, "node_modules"), path.join(dir, "node_modules"), "dir");
  return dir;
}

/** Paths `npm pack` would publish from `dir`, running lifecycle scripts (prepare, prepack). */
export function packedFiles(dir: string): string[] {
  const out = execFileSync("npm", ["pack", "--dry-run", "--json", "--ignore-scripts=false"], {
    cwd: dir,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  // Script output can precede the JSON report.
  const report = JSON.parse(out.slice(out.indexOf("["))) as Array<{ files: Array<{ path: string }> }>;
  return (report[0]?.files ?? []).map((file) => file.path);
}

/** Builds src/ with tsconfig.build.json into `outDir` and returns every emitted file. */
export function buildInto(outDir: string): string[] {
  const tsc = path.join(repoRoot, "node_modules", "typescript", "bin", "tsc");
  execFileSync(process.execPath, [tsc, "-p", path.join(repoRoot, "tsconfig.build.json"), "--outDir", outDir], {
    cwd: repoRoot,
    stdio: "pipe",
  });
  return walk(outDir);
}

/** True when a package-relative path is covered by package.json "files" (or always shipped). */
export function isPublished(pkgRelPath: string): boolean {
  const pkg = JSON.parse(readFileSync(path.join(repoRoot, "package.json"), "utf8")) as { files?: string[] };
  const p = pkgRelPath.split(path.sep).join("/");
  if (p === "package.json") return true;
  return (pkg.files ?? []).some((entry) => {
    const e = entry.replace(/^\.\//, "").replace(/\/$/, "");
    return p === e || p.startsWith(`${e}/`);
  });
}

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    return statSync(full).isDirectory() ? walk(full) : [full];
  });
}
