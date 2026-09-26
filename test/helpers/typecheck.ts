/**
 * Type-level test support. Vitest does not typecheck, so public-type
 * regressions are caught by running the TypeScript compiler over in-memory
 * snippets with the project's tsconfig.
 */

import path from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const testDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const root = path.resolve(testDir, "..");

/**
 * Compiles every snippet in one program and returns each one's diagnostics
 * (flattened messages, empty when it typechecks). Snippets are virtual files
 * in test/, so they import the package as "../src/index.js". `overrides`
 * adjusts the project's compiler options, e.g. to compile as a consumer with
 * stricter settings would; only the snippets' own diagnostics are returned.
 */
export function typeErrors(
  snippets: Readonly<Record<string, string>>,
  overrides: ts.CompilerOptions = {},
): Record<string, string[]> {
  const config = ts.readConfigFile(path.join(root, "tsconfig.json"), (p) => ts.sys.readFile(p));
  const parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, root);
  const options: ts.CompilerOptions = { ...parsed.options, ...overrides, noEmit: true };
  const files = Object.entries(snippets).map(([name, text]) => ({
    name,
    text,
    fileName: path.join(testDir, `__typecheck_${name}__.ts`),
  }));
  const virtual = new Map(files.map((file) => [file.fileName, file.text]));

  const host = ts.createCompilerHost(options);
  const getSourceFile = host.getSourceFile.bind(host);
  host.getSourceFile = (fileName, languageVersion, onError, shouldCreate) => {
    const text = virtual.get(path.resolve(fileName));
    return text === undefined
      ? getSourceFile(fileName, languageVersion, onError, shouldCreate)
      : ts.createSourceFile(fileName, text, languageVersion, true);
  };
  const fileExists = host.fileExists.bind(host);
  host.fileExists = (fileName) => virtual.has(path.resolve(fileName)) || fileExists(fileName);

  const program = ts.createProgram([...virtual.keys()], options, host);
  return Object.fromEntries(
    files.map(({ name, fileName }) => {
      const sourceFile = program.getSourceFile(fileName);
      // Without a source file, getPreEmitDiagnostics would report the whole program.
      if (sourceFile === undefined) throw new Error(`typeErrors: ${fileName} was not compiled.`);
      const messages = ts
        .getPreEmitDiagnostics(program, sourceFile)
        .map((d) => ts.flattenDiagnosticMessageText(d.messageText, "\n"));
      return [name, messages];
    }),
  );
}
