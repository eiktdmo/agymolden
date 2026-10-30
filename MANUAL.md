# newBRORSEN v04.00-v04.09 - Technical Manual

> **Two supported lines (2026-10-20).** **v04.09** is the wrapper for the new library, pyscf_neo_cc
> 0.1.0.post20261008 (env `MCCC_1008_stock`). It is a development build until signed off. **v04.08** is kept as the
> legacy-support version for the 2026-09-23 library (env `MCCC_0923_stock`). Its manual and user guide are archived
> as `archive/MANUAL.v04.08.md` and `archive/USER_GUIDE.v04.08.md`. v04.09 does not run on the 2026-09-23 library,
> and v04.08 is not adapted to the new one: on it, two-positron DF energies silently switch reference.

> **Running calculations?** Start with `USER_GUIDE.md`: the same options explained for users, without the
> development history. This manual is the complete reference (changes, library details, measurements, tests).

Unified multicomponent coupled-cluster wrapper for **pyscf + pyscf_neo_cc**: electrons,
positrons (one or two) and quantum protons; single points, gradients, geometry optimizations
and potential-energy scans; multi-stage chained jobs.

> **Status: v04.06 is the production release** (signed off 2026-10-06; `TEST_REPORT.md` section 18). It is v04.05
> (production 2026-09-30 to 2026-10-06) plus a Molden / cube file viewer. Use it with pyscf_neo_cc 2026-09-23 on stock PySCF 2.14
> (environment `MCCC_0923_stock`). v04.00-v04.03 were never released; v03.17 localfix remains the reference for
> results made before the switch to the new library. Inputs written for v04.04 may need `migrate_v0405_inputs.py`.
> A release sets `RELEASED = True` near the top of the wrapper file.
> v04.02 and later are meant for pyscf_neo_cc 2026-09-23 on stock PySCF 2.14 (conda environment
> `MCCC_0923_stock`); on the older library some v04.04 features are refused with a clear error.
>
> **v04.01** is v04.00 plus the gradient-free Z-matrix optimizer (`%opt_engine zopt`, section 10.1) and
> the close-atom check (section 15). It is kept in its own file until ZOPT has been tested
> as thoroughly as the rest; everything else behaves exactly as in v04.00.
>
> **v04.02** is v04.01 adapted to the fitting-set roles of pyscf_neo_cc 2026-09-23: an RI set
> never reaches the mean field (an RI set given as `%jkfit_aux_e` is moved to the correlation
> role, with a warning), and CONV calculations use an exact mean field (section 7.1, "Fitting
> sets and their roles"). Tested on both the old and the new pyscf_neo_cc.
>
> **v04.03** is v04.02 with measured fitting-set defaults: the mean field uses the JK-fit set
> matched to the electron basis (`aug-cc-pvtz-jkfit` for aug-cc-pvtz), and the electronic sector
> fits its correlation with the matched RI set (section 7.1).
>
> **v04.04** is v04.03 with the optimization results (final geometry, DF energy, summary) written
> *before* the `%conv_check` exact-integral single point, so a job killed for memory during that
> check keeps its results; it also takes **particle masses**: antimuons, protons, deuterons,
> tritons and muonic helium (`%pos_mass`, `%quantum_site`, `%nuc_mass`; section 7.4). Added to the same
> file on 2026-09-24/26, version number kept:
> - ghost atoms (section 6.3);
> - MP2 in all sectors (section 7.5);
> - constrained NEO (`CNEO`, `CDFT`; section 7.3), with fixes to the PROTON sector;
> - hybrid ZOPT with optional analytic gradients (section 10.1);
> - the two-positron fitting set chosen per element, plus a pre-flight check of the DF gradient
>   (section 7.1).
>
> **v04.08** (development build, not released; 2026-10-14) is the v04.07 development build renamed by the
> user, plus atoms on the basis line (`%pos_basis PSX-DZ H PSX-TZ {1 2}`, `%basis aug-cc-pvtz H cc-pvdz`;
> section 8.2), and new input checks (unknown
> `%basis` names, a missing `%next`). Engine `newBRORSEN.v04.08.py`, input builder `input_builder.v04.08.py`.
>
> Full list: section 19.

Files in `wrapper/unified/`:

| File | What it is |
|---|---|
| `newBRORSEN.v04.00.py` | the wrapper (single file) |
| `newBRORSEN.v04.01.py` | v04.00 + the ZOPT optimizer (`%opt_engine zopt`, section 10.1) |
| `newBRORSEN.v04.02.py` | v04.01 + fitting-set roles for pyscf_neo_cc 2026-09-23 (section 7.1) |
| `newBRORSEN.v04.03.py` | v04.02 + matched fitting-set defaults (section 7.1) |
| `newBRORSEN.v04.04.py` | previous production release (2026-09-26 to 2026-09-30): v04.03 + results saved before the `%conv_check` single point, particle masses, ghost atoms, MP2, CNEO/CDFT, hybrid ZOPT, per-element two-positron fitting set and pre-flight check (section 19) |
| `newBRORSEN.v04.06.py` | **production release** (2026-10-06): v04.05 + a Molden / cube file viewer (`python newBRORSEN.v04.06.py file.molden`) |
| `newBRORSEN.v04.05.py` | previous production release (2026-09-30 to 2026-10-06): v04.04 + electronic geomeTRIC, FREQ, TS/IRC/NEB, analyses (CHELPG, maps, Molden, cube), `--resume`, new defaults and input rules (section 19) |
| `newBRORSEN.v04.07.py` | previous development build (2026-10-08 to 2026-10-14): v04.06 + the CLI flags (`--view`, `--parse-text-json`) the input builder needs |
| `input_builder.v04.07.py` | previous development build of the input builder |
| `newBRORSEN.v04.08.py` | **development build** (not released; 2026-10-14): v04.07 + atoms on the `%basis` / `%pos_basis` line, input checks (section 19) |
| `input_builder.v04.08.py` | development build: the terminal input builder, its own program next to the engine (section 2.1) |
| `archive/make_input.py` | v04.05 question-and-answer input maker, retired in v04.07 (replaced by the input builder) |
| `migrate_v0405_inputs.py` | rewrites v04.04 inputs for v04.05 (new copies; the originals are not changed) |
| `template.inp` | every option in one commented, runnable input (quick reference) |
| `MANUAL.md` | this manual |
| `tests/` | pytest suite (section 18) |

---

## Contents

1. What v04.00 is
2. Running it
3. Input file structure
4. The `!` line
5. Directive reference
6. Geometry input (Cartesian, Z-matrix, ghost atoms)
7. Sectors: POSITRON, ELECTRONIC, PROTON and constrained NEO (7.3); other particles (7.4); MP2 (7.5)
8. Positron basis sets and atom selection
9. Run types
10. Geometry optimization
11. Constraints, pinnings and equivalences
12. Potential-energy scans, Morse and harmonic fits
13. Output files
14. Resources, memory and clusters
15. Validation: errors and warnings
16. Known library limitations and workarounds
17. Migrating from v03.17 and from twoposBrorsen
18. Tests
19. Changelog

---

## 1. What v04.00 is

v04.00 merges two wrapper lineages into one file:

- **newBRORSEN v03.17** (`_localfix.py` / `_merged.py`): the wrapper used for this
  project's production runs. Its computational code (SCF, CC, triples, gradients,
  geomeTRIC optimizer, built-in WOPT optimizer, one-line Morse scan, final-energy handling) is
  carried over unchanged. Section 18 shows it gives the same numbers as `_merged.py`.
- **twoposBrorsen 00.44-00.86** (and the older Brorsen-MCCCSDT-v5 wrapper, its ancestor):
  its input syntax, Z-matrix input, polyatomic and Z-matrix scans, `.scan` tables, triplet
  positron pairs, `%df_aux`, SCF convergence fallback and coordinate pinnings/equivalences.

Any input written for either wrapper runs. Where the two disagreed on meaning, v04.00 keeps
the behavior that is correct and says so in the output (sections 15 and 17). Several bugs in
both originals are fixed (section 19).

## 2. Running it

```bash
python newBRORSEN.v04.06.py job.inp            # run all stages
python newBRORSEN.v04.06.py job.inp --dry-run  # parse + validate + print the stage summary, compute nothing
python newBRORSEN.v04.06.py job.inp --decimal-comma  # decimal comma in report/table files
```

Use the Python of the `MCCC_0923_stock` environment (`~/.conda/envs/MCCC_0923_stock/bin/python`) for
v04.02 and later.

`--decimal-comma` writes every number in `<job>.scan`, `<job>_scan_points.txt`,
`<job>_morse_fit_summary.txt`, `<job>_harmonic_fit_summary.txt`, `<job>_summary.txt` and
`<job>_profile.txt` with a comma
(`-460,2376091493`), for spreadsheets set to a Spanish-style locale. The values are identical;
only the separator changes. `.xyz` files and the `.out` log keep dots (other programs read
them), and input files always use dots (commas separate values there, e.g. `%pin 1,2`).

- Everything printed also goes to `job.out` (written and flushed line by line, so it
  survives a crash or a killed job).
- Output files are written to the **current directory**. Run each job in its own folder.
- Do **not** run from the `brorsenMCCC/` root: the `pyscf/` source checkout there shadows
  the installed package and imports fail. Any other directory is fine.
- With the Singularity image, keep the wrapper outside the image:
  `singularity exec brorsen_mcccsdt.sif python3 /path/to/newBRORSEN.v04.05.py job.inp`.

**Resuming (v04.05).** For cluster jobs that SLURM may requeue.
- Inside a SLURM job (`SLURM_JOB_ID` set), every stage writes `<job>_stage<N>_started.json` when it begins and
  `<job>_stage<N>_done.json` when it finishes. The done file holds the stage's result: final geometry, energy
  or gradient. The log says when these stage records are on.
- Local runs write no JSON files: only the plain-text results and `<job>_checkpoint.xyz`, as before. To test a
  resume locally, add `--stage-records` (or run with `--resume`, which also writes them).
- `--resume` skips every stage whose done file matches the input and uses its stored result for the next
  stage.
- A stage "matches" when every setting it runs with (including those carried over from the stage before) and its
  start geometry are unchanged. An edited stage, a stage whose carried-over settings changed because an earlier
  stage was edited, or one whose start geometry changed, runs again. Comments and layout do not count.
- An optimization that started but did not finish restarts from the last geometry in
  `<job>_checkpoint.xyz`.
- ZOPT already restarts from its history file.
- After a SLURM requeue (`SLURM_RESTART_COUNT` > 0), `--resume` is automatic.
- On resume the `.out` log is appended to, not overwritten.

Without `--resume` and outside a requeue, every stage runs as before.

### 2.1 Input builder: `input_builder.v04.08.py` (v04.07; v04.08)

```bash
python input_builder.v04.08.py            # start screen: New input, Open, Recent files, View molden/cube, Settings, Quit
python input_builder.v04.08.py job.inp    # open (or create) job.inp directly
python input_builder.v04.08.py job.molden # straight into the Molden/cube viewer (run_viewer), not the editor
```

A terminal editor for input files, split out of the engine wrapper into its own program (next to it) so it
starts without importing numpy/PySCF. It needs `prompt_toolkit` in the environment
(`python -m pip install prompt_toolkit`) and a terminal of at least 80x24. It starts in well under a second.
It drives the engine named in Settings > **Engine path** (auto-detected when a matching-version
`newBRORSEN.vNN.NN.py` sits next to it) through that script's own `--dry-run`, `--parse-text-json` and
`--view` command-line flags, and, for the Molden/cube viewer only, by loading that script's module
namespace lazily with `runpy` the moment the viewer is opened.

- **New input** asks for the geometry first: type the atoms one by one, paste xyz rows, or load a
  `.xyz`/`.zmat` file (the same menu F2 opens later; Esc leaves the blank template with its gray example
  geometry instead, editable with F2 whenever). This step is skipped when a geometry is already given
  another way: the command-line geometry argument, a `.xyz`/`.zmat` file argument, or **Open** on an
  existing `.inp`. Given on the command line, rows separated by newlines or `;`:

  ```bash
  python input_builder.v04.08.py "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
  python input_builder.v04.08.py "* xyz -1 2; F 0 0 0; H 0 0 0.92; *"
  python input_builder.v04.08.py "Li; H 1 r; r = 1.6"          # a Z-matrix
  python input_builder.v04.08.py molecule.xyz                  # or molecule.zmat
  ```

  In each case: the template, with that geometry already in place as an `* xyz` or `* zmat` block. Charge
  and multiplicity come from the file when it has them: an `.xyz` comment line `0 1` or `charge=-1 mult=2`,
  or a `* xyz|zmat <charge> <mult>` header. Otherwise the geometry menu asks for them, suggesting the
  multiplicity from the electron count. Only the description section between the `# ====...` lines and the
  geometry are real at this point; the rest of the template is gray, which means it is not part of the
  input yet:
  - a `!` line with every word: run types, sectors, methods and integrals. The units are not offered there;
    they are an option, `%coords angstrom|bohr`. `! Angs` / `! Bohrs` still work in inputs that have them;
  - `%basis`, `%memory` and `%nprocs` start real (their usual default) instead of gray: every run needs
    them. `%jobname` is also real from the start, auto-generated as `formula_method_q<charge>_m<mult>_basis`
    (e.g. `H2O_HF_q0_m1_ccpvdz`, from the Hill-system formula of the geometry, the `!` method, the stage's
    charge/mult and `%basis`) and keeps itself current as any of those change — until the line is edited by
    hand, after which it is left alone for good;
  - the remaining options, one gray line each, with their default or an example value and a short
    description, and block options with their syntax rows.

  Once a geometry and a `%basis` are both present, a `%basis` that does not cover every element in the
  geometry (checked against the installed basis set's own data file, no calculation needed) is flagged
  yellow (`W`), naming a basis that does cover them when `def2-tzvp` or `def2-qzvp` does.

  **On the `!` line**, move the cursor onto a gray word and press **Tab**. That word becomes real, and the
  other gray words of its group disappear: choosing OPT removes SP, GRADIENT, TS and SCAN but keeps FREQ,
  because `OPT FREQ` is valid. On gray text Tab always means "keep this" (an open completion list does not
  take it). The Right arrow keeps moving the cursor, so that any word can be reached.

  **Options appear only where they apply.** Choosing the run type, sector, method or integrals removes the
  options that cannot apply. Examples: positron options under ELECTRONIC, OPT-only options under SP, `%xc`
  unless DFT, `%scan` unless SCAN, fitting sets and `%conv_check` under CONV (both need a DF calculation to
  check or to fit), and `%compute_eee` / `%compute_t2e1p` unless CCSD(T). Options that depend on another one
  appear once that one is kept:
  - `%ts_mode` is offered as three gray lines (ts, irc, neb). Tab on one keeps it and removes the other two;
    `neb` brings in `%neb_product`, `%neb_images` and `%neb_optimize_ends`, and `irc` brings in
    `%irc_direction`;
  - `%opt_engine` is one line (AUTO). Its value completes like a basis set, from a list with what each engine
    means (AUTO, GEOMETRIC, WOPT, ZOPT). Writing ZOPT brings in the `%zopt_*` settings right away; a changed
    value updates the options offered without another Tab;
  - `%scan` brings in `%scan_fit`, `%cube true` brings in `%cube_points`, `%density_map` brings in
    `%density_map_plane`, `%n_pos` brings in `%pos_mult` / `%pos_spin` (any value, not just 2: n_pos 3 with
    pos_mult 4 and the like are real, supported combinations), `%quantum_nuc` brings in `%nuc_mass`, and
    `%cneo true` (or CNEO) brings in `%cneo_variant`.

  **Tab on a gray option line** keeps it (a whole block at once). Editing a gray line also makes it real.
  Deleting a gray line (vim `dd`) removes it for good from this stage's template.

  **Undo** (`u` / Ctrl-R in vim, `C-_` in emacs) also undoes and redoes these choices, including which words
  and lines are gray.

  **`%next` shows the template of the next stage.** When `%next` is accepted (it is suggested on the empty
  line after a finished stage, also at the end of an opened file) with **Tab or Enter**, the stage above
  keeps only what was chosen (its gray template stops being offered) and the new stage gets its own template. In it, the `!` words and option values of the stage before
  are **highlighted** in cyan (italic, underlined) but not accepted. All the run-type words it used count
  (`OPT FREQ` highlights both). **Shift-Tab**, or **Tab Tab** (two quick Tabs), keeps every highlighted word
  and option line at once. The first Tab of the pair is taken back, so the result is exactly the stage
  before; then change what differs. Following the wrapper's inheritance rules,
  a stage without a geometry block continues the one before: what is left highlighted is inherited anyway,
  and choosing another word or keeping a line with a new value changes it for this stage. The new stage has
  no example geometry; F2 adds one if the stage needs a new geometry.

  **Saving** writes only the real lines and then removes the gray ones from the editor. The checks while
  typing ignore gray text. With every line still offered kept, the template gives a valid input for every
  run type tested (`tests/inputs/ui_template/`).
- **Search:** vim `/` and `?` (`n` / `N` for the next match), emacs `C-s` / `C-r`, plain Ctrl-F. In emacs
  `C-s` is search, so it does not save there (`C-x C-s` does).
- **Gray suggestions.** While you type, the rest of a common line appears in gray, as in a shell:
  - after `!`, the `!` combinations used most often in `results/` (`UI_COMMON_BANG`);
  - after `%` and a few letters, the key and its value;
  - after `%key `, its value. The value is the wrapper's default (`%memory 4000`), or an example of the
    syntax for options without a literal default (`%pin 1,2`, `%equivalence 1,2,3,4`, `%pos_basis PSX-TZ 1 2`,
    `%scan 2.0 4.0 9`). On/off options get the value that is not the default (`%molden true`);
  - after a block option is completed with Tab (e.g. `%cons` + Tab gives `%constraints`), its syntax: a gray
    example row (`bond 1 2 1.10  # bond i j [A]; angle i j k [deg]; ...`) and a real `end`. Tab on the gray
    row keeps it as a start, or replace it with your own rows;
  - `%next`, on an empty line after a finished stage (its `!` line and a closed geometry block; for example
    right after the geometry builder).

  The Right arrow or Tab accepts the suggestion (in every keymap; Tab steps through the list instead while the
  completion list is open), and typing on ignores it.
- **Basis sets.** After `%basis`, `%pos_basis`, `%nuc_basis`, `%jkfit_aux_e` or `%auxbasis_e`, the list offers the
  sets that the installed libraries ship. The list comes from PySCF's basis table (about 370 names, common ones
  first; `%pos_basis` adds PSX-DZ/TZ/QZ/5Z and MATCH) and from pyscf_neo_cc's nuclear bases (`pb4d`, `pb4f1`,
  ..., `pb6h`). Fitting options get the jkfit and ri sets. The files are read directly, so the input builder
  does not import PySCF. Dashes do not matter when typing (`augccpvq` finds `aug-cc-pvqz`).
- **Typing.** `!` opens the keyword list. Words from a group already on the line (run type, sector, method,
  integrals, units) are no longer offered, except FREQ after OPT or TS. `%` opens the directive list, with each
  default and a one-line description. Directives of the current sector come first; the others are dimmed.
  After `%key ` the allowed values are offered. In geometry rows, element symbols are offered.
- **Checks while typing.** Conflicting `!` words, removed words, unknown directives, bad values, positron or
  proton directives without their sector, a ZOPT without a `* zmat` block, a NEB without `%neb_product`,
  decimal commas in coordinates, and an electron count that does not fit the multiplicity are drawn in red,
  and warnings in yellow. The left margin shows `E` or `W`, and the status line gives the message for the
  cursor's line (on the stage's `!` line, for a missing option — there is no line of its own to point at).
  Colors are the terminal's own 16 ANSI colors.
- **Basis-set options are a yellow reminder (`W`), not an error, if missing**, once a sector has been chosen
  (nothing is flagged before that): `%basis` (any sector, default cc-pvdz), POSITRON's `%pos_basis` (default
  PSX-TZ), PROTON's `%nuc_basis` (default pb4f1) and `%quantum_nuc` (default `0`, the first atom). Each one
  does have a working default, so this is a nudge to set it on purpose, not a block. While its gray template
  line is still offered, the margin shows `W` there directly; once typed over or no longer offered (an
  opened file that never had it), the same `W` moves to the stage's `!` line instead.
- **F3 (or vim `:done`) stops offering this stage's template right now**, the same as saving would (every
  gray part is dropped), without saving — for when you are done choosing and do not want to keep seeing
  suggestions while editing something else.
- **F4 shows hidden `%` options in the completion menu too** — they still validate when typed either way; F4 only controls whether
  they are suggested, dimmed when shown. Press F4 again to go back to hiding them.
- **Stages.** In a stage after `%next` that has no geometry block, the status line shows what it inherits
  (run type, method, sector, basis). `%next` is offered once the stage has a geometry.
- **Geometry (F2 or Ctrl-G).** The builder works in the bottom line of the editor, not in a dialog. Pick
  with the arrows and Enter; Esc cancels. Wherever F2 is pressed, the geometry goes into the stage's geometry
  block after its options. The gray example, or a geometry already there, is replaced, and one undo brings
  it back. A stage without one gets it after its last line.
  - **Type atoms:** asks how many atoms, then each atom's label (with element completion; `C1`, `ghost-H`
    and `H+` are accepted) and its `x y z` in Angstrom. Each atom appears in the `* xyz` block as it is
    entered.
  - **Paste xyz rows:** inserts an empty `* xyz` block. Paste the rows (an `.xyz` file's count and comment
    lines may be pasted too) and press F2 again: the rows are cleaned up and lines that are not
    `Element x y z` are dropped.
  - **Load .xyz / .zmat file:** the first matching file or folder name appears in gray; Tab or the Right
    arrow completes it. Folders end in `/`, so completing one goes straight into it. The Open and Save
    dialogs complete names the same way. A Z-matrix stays a `* zmat` block, so named variables
    still work for ZOPT and `sym_zmat` scans.
  - **Continue stage N-1:** in stage 2 and later, no geometry block is needed.

  It then asks the charge (default: the stage's `%charge`) and the multiplicity (default: suggested from
  the electron count) and writes them in the `* xyz` header. It finishes in vim's normal mode, on an empty
  line after the block where `%next` is suggested.
- **View molden/cube** (start screen, before Settings; also `input_builder.v04.08.py job.molden` / `job.cube`
  directly, or `newBRORSEN.v04.08.py --view job.molden [file2]` without the input builder at all, for the
  older menu described under v04.06 in the changelog). A native screen, split left/right: the left pane lists loaded files and settings
  (Up/Down to move, Left/Right to cycle a value, Enter/Space to act on the selected row), the right pane
  previews the last plot. Rows:
  - loaded files, each `[x]`/`[ ]` (included in **Compare**, below) with its label, species role and kind;
    Enter toggles inclusion, `r` renames the label, `d` removes it;
  - **+ Add file...** asks for a `.molden`/`.cube` file, then (unless it is the file the screen opened with)
    whether it is the electron density, the positron density, or other (comparison only);
  - **Map** (electron / positron / both / esp), **Plane**, three **Colour range** rows, **Colour scheme**,
    **Y-axis labels every**, **Output prefix** — the same settings v04.06's menu offered;
  - **Draw**: the electron/positron/both/esp map or CHELPG-style request above, from the most recently loaded
    electron-role and positron-role files (same behaviour as v04.06's menu);
  - **Compare included files**: one panel per `[x]` file (any role), each its own label, side by side in one
    figure — for comparisons the fixed electron/positron pairing does not cover, for example the electron
    density of the same system with 0, 1 and 2 positrons attached, each file loaded and labeled accordingly;
  - **CHELPG charges**, **Files info**, **Close** (also Esc or `q`).

  Numpy and PySCF, which this screen needs (`CubeGrid`, Molden reading, the plane-map plotting) and the
  input builder never imports on its own, are loaded from the configured engine script the moment this
  screen is opened (about 1 s), not before: plain typing sessions that never touch it start in well under a
  second. `input_builder.v04.08.py job.molden` opens straight into it, the file already loaded (guessing its
  role; no prompt); from inside the screen a role is always asked.

  **Preview pane** is entirely optional, and only appears when it can show something: Kitty graphics
  protocol (Kitty itself, WezTerm, Ghostty; no extra tool needed) or iTerm2's inline images, else Sixel
  (`foot`, Konsole) if ImageMagick (`magick`/`convert`, already used for the plane maps themselves) or
  `img2sixel` is installed, else plain ANSI/Unicode block-character art via `chafa` if that is installed
  (works in any colour terminal, no special protocol needed). All of this is checked, not guessed, at
  startup (also correctly over SSH, since `TERM`/`TERM_PROGRAM` are the client's own). With none of the
  above, the screen is just the file/settings pane at full width, nothing missing from it: Draw and Compare
  still write the plot files, opened in the system's own image viewer (`xdg-open` / macOS `open`) instead
  of previewed inline.
- **Dialogs and menus** (save, run, settings, recent files): the arrows move between choices and fields,
  Enter picks, and Esc cancels. Tab also works.
- **Help (F1, or `?` in vim normal mode).** The MANUAL row of the directive under the cursor, section 4 for a
  `!` word, or section 6 for a geometry line. Esc, F1 or q closes it.
- **Keymaps** (Settings: Up/Down choose a setting, Left/Right change its value, Enter closes):
  - `vim` (default; starts in normal mode): `:w`, `:w name.inp`, `:wq`, `:q` (back to the start screen),
    `:q!` (back without saving), `:qa` (leave the input builder), `:geom`, `:help`, `:start`, `:done` (stop
    offering this stage's template); `u` / Ctrl-R undo and redo;
  - `emacs`: `C-x C-s` saves, `C-x C-c` goes back to the start screen, `C-s` / `C-r` search, `C-_` undoes;
  - `plain`: arrows, `Ctrl-S` saves, `Ctrl-Q` goes back to the start screen, `Ctrl-F` searches.

  Mouse support can be switched off. Settings and the last five files are kept in
  `~/.config/newbrorsen/ui.toml`.
- **Saving.**
  - New files go to the current folder, named after `%jobname` by default.
  - The first save of an opened file asks: replace it, or save under a new name in the same folder
    (`name_2.inp` offered).
  - Names may use letters, digits, `.`, `_` and `-`, and must not start with `-`.
  - The text is written exactly as it is in the editor: comments, case and aliases are never rewritten.
  - Before saving, the wrapper's own parser and validator check the text (`--parse-text-json`, about 2 s,
    writes no files). Errors are listed with their lines, and saving anyway needs a confirmation.
- **After saving.**
  - **Dry-run** runs `--dry-run` on a copy in a temporary folder, so the job folder gets no `.out`, and shows
    the stage summary.
  - **Run in background** starts `python -W ignore newBRORSEN.v04.08.py job.inp --decimal-comma` in the
    input's folder. It is detached and started without a shell, with output in `nohup.out`, the PID in
    `pid.txt` and the log in `job.out`.
  - If the folder already holds another `.inp`, `run.log`, `rc.txt`, `job.out` or a running job, the UI
    offers to move the input into `./job/` first. It never overwrites a job.
  - A second launch in a folder whose `pid.txt` process is alive is refused.
  - After the launch: view the `.out` (follows new output; Esc or q goes back), keep editing, start a new
    input, or return to the start screen.

The input builder's checks use their own option table (`UI_OPTIONS`, `UI_BANG`, in `input_builder.v04.08.py`).
`tests/test_ui.py` checks it against the engine's parser (`newBRORSEN.v04.08.py`): every directive the
parser knows is in the table, every table entry and alias is accepted, and the checker reports no error on
any input in `tests/inputs/` that the wrapper accepts. A new directive therefore needs a `UI_OPTIONS` row,
or that test fails.

Always start with `--dry-run`. It shows every setting each stage will use, the resolved
frozen-core count, the atoms that carry positron functions, the positron basis actually
used, and every warning, before any CPU time is spent.

## 3. Input file structure

```
! <run type> <method> <sector> <integrals>      <- stage 1
%directive value
* xyz <charge> <mult>
<atoms>
*
%next                                            <- stage separator (%newjob also works)
! ...                                            <- stage 2
...
```

- **Stages** run in order. A stage with no geometry block continues from the previous
  stage's result: the optimized geometry after `OPT`, the lowest-energy point after any scan
  (v04.05; up to v04.04 a fit passed its fitted minimum), otherwise the previous stage's input geometry.
- **v04.05 (2026-09-30): a stage without a geometry block also continues all the settings of the stage
  before** (method, sector, integrals, run type, basis sets, charge, multiplicity, positron/proton options,
  resources, analyses, constraints). What the stage writes replaces them: a `!` word replaces its group, a
  directive its value. Not continued: `%jobname`, the geometry-block settings (units, Z-matrix), `%scan` and
  `%neb_product` blocks, and the SCAN / NEB run types that need them (such a stage becomes SP_ENERGY). A stage
  with a geometry block is a new calculation and starts from the defaults. Up to v04.04 only the geometry
  was inherited; none of the 945 project inputs changes meaning (their 199 geometry-less stages repeat every
  setting). `--dry-run` prints `Settings: continued from stage N` for such stages.
- Validation (and `--dry-run`) checks that each stage's charge and multiplicity fit its number of
  electrons.
- **Comments:** `#` starts a comment anywhere on a line (a `#` inside quotes is kept).
- **Directives:** `%key value`, `%key=value` and `%key = value` are equivalent. Keys are
  case-insensitive. An unknown directive or `!` token prints a warning (it used to be silent).
- **Blocks** (`%constraints`, `%pinnings`, `%equivalences`, `%scan`) are
  closed by a line `end`. A block left open is closed at the next `%`, `!` or `*` line with a warning.
- **Atom numbers are 1-based** everywhere (first atom = 1). The one exception is
  `%quantum_nuc`, which keeps the 0-based PySCF convention. See the table below.
- **Units:** Hartree, Angstrom, degrees. `%memory` in MB, `%df_mem_cap` in GB.

### 3.1 Atom numbering

Atoms are numbered 1, 2, 3, ... in the order they are listed in the geometry block (`* xyz`
or `* zmat`, where row *n* is atom *n*). The same numbers are used by every option:

| Where atom numbers appear | v04.00 | v03.17 (`_localfix`/`_merged`) | twoposBrorsen |
|---|---|---|---|
| Z-matrix references (`H 1 0.96`) | **1-based** (or a row label) | no Z-matrix | 1-based |
| `%constraints` | 1-based | 1-based | 0-based in its code (a bug) |
| `%pin`, `%equivalence a,b,c,d` | 1-based | 1-based | - |
| `%pinnings` / `%equivalences` blocks | 1-based | ignored | 1-based (early 00.03-00.16 test inputs wrote 0) |
| `%scan bond i j`, `translate i k` | 1-based | - | 1-based |
| `%quantum_nuc` (PROTON) | **0-based** (passed straight to PySCF) | 0-based | - |

A `%pinnings`, `%constraints` or `%equivalences` block containing atom `0` can only be 0-based,
so v04.00 reads that block as 0-based and warns (section 11.7).

## 4. The `!` line

One word from each group, in any order. Missing groups take the default.

| Group | Accepted words | Default |
|---|---|---|
| Run type | `SP_ENERGY` (`SP`, `ENERGY`), `GRADIENT` (`GRAD`, `SP_GRAD`), `OPT` (`GEOMOPT`, `OPTIMIZATION`; the optimizer is set with `%opt_engine`), `SCAN` (fit: `%scan_fit`), `FREQ` (`FREQUENCIES`, `HESSIAN`), `ANFREQ`, `NUMFREQ`, `TS` (`OPTTS`; `%ts_mode ts|irc|neb`) (v04.05) | `SP_ENERGY` |
| Method | `HF`, `DFT`, `MP2` (v04.04, section 7.5), `CCSD`, `CCSD(T)` | `HF` (v04.05, 2026-09-30; `CCSD(T)` before) |
| Sector | `POSITRON` (`POS`), `ELECTRONIC` (`ELEC`), `PROTON` (`NEO`); `CNEO` = PROTON with `%cneo true`; `CDFT` = `DFT CNEO` (v04.04, section 7.3) | `ELECTRONIC` (v04.05; `POSITRON` before) |
| Integrals | `DF` (`RI`), `CONV` (`EXACT`) | `DF` |
| Coordinate units (v04.05) | `Angs` (`Angstrom`, `Ang`), `Bohrs` (`Bohr`, `AU`), as in ORCA; the same as `%coords` | `Angs` |

twoposBrorsen's positional form `! OPT CCSD POSITRON DF` is simply these four words.

- v04.05 (director's request, 2026-09-30): the optimizer is not chosen on the `!` line. Write `! OPT` and
  `%opt_engine wopt` (built-in BFGS), `%opt_engine geometric` (the default) or `%opt_engine zopt`
  (section 10.1). `%opt_engine zopt` in a stage that is not `OPT` is ignored with a warning.
- A `!` line without a run type runs `SP_ENERGY` (v04.05: no warning; twoposBrorsen defaulted to `OPT`).
- Two words of one group on the `!` line are an error (v04.05, 2026-09-30); before, the last word won silently
  (`! HF CCSD` ran CCSD, `! OPT FREQ` only FREQ). The exception, as in ORCA: `! OPT FREQ` (either order; also
  `NUMFREQ` / `ANFREQ`) is split into an OPT stage and a FREQ stage at its result, with the same settings and job
  name (the FREQ files `<job>_freq.*` do not clash with the OPT ones; stages after it are numbered on). `! TS FREQ`
  is accepted and is the TS stage with its frequency check (`%ts_freq true`, the default).
- A `!` line without a sector runs ELECTRONIC (v04.05; POSITRON up to v04.04). Positron directives (`%n_pos`,
  `%pos_basis`, ...) or proton directives (`%quantum_nuc`, `%nuc_basis`, ...) in a stage that does not name
  `POSITRON` or `PROTON` are an error, so an old input that relied on the POSITRON default cannot silently
  become an electronic calculation. An unknown `!` word is an error.
- `DFT` exists for the ELECTRONIC and PROTON sectors only.

## 5. Directive reference

### 5.1 General and resources

| Directive | Aliases | Value | Default | Notes |
|---|---|---|---|---|
| `%jobname` | | text | `Stage_N` | prefix of every output file of the stage |
| `%charge` | | integer | 0 | electronic charge (positrons are not counted) |
| `%coords` | `%coord_units`, `%units` | `bohr` (`bohrs`, `au`) or `angstrom` (`angs`); ORCA's `%coords units bohrs end` on one line also works | `angstrom` | v04.05: units of the coordinates written in this stage's `* xyz` / `* zmat` block and `%neb_product` (section 6.4) |
| `%mult` | `%multiplicity` | 2S+1 | 1 | |
| `%spin` | | 2S | 0 | sets `%mult` = spin + 1 |
| `%nprocs` | `%threads`, `%cores` | integer | 4 | threads used by PySCF; set per stage |
| `%memory` | | MB (`16GB` accepted) | 4000 | PySCF memory target; also applied to the positron SCF's initial guess |
| `%df_mem_cap` | | GB | library default: min(32 GB, half the RAM) | pyscf_neo_cc's byte cap for blocked DF intermediates; reset to the default in stages that do not set it |
| `%verbose` | | integer | quiet | PySCF verbosity of the SCF/CC objects |

### 5.2 Electrons

| Directive | Aliases | Value | Default | Notes |
|---|---|---|---|---|
| `%basis` | `%e_basis` | basis name, or `B1 <atoms> B2 <atoms> ... [B]` (v04.08) | `cc-pvdz` (v04.05; `aug-cc-pvtz` before) | several bases by element or atom number; every atom needs one (a basis with no atoms after it covers the rest); section 8.2 |
| `%frozen` | | `auto` or integer | 0 | `auto` = chemical core: 0 for H-He, 1 orbital for Li-Ne, 5 for Na-Ar, 9 for K-Kr, summed over atoms. Applied in every sector and engine (section 16 for the two-positron caveat) |
| `%jkfit_aux_e` | | basis name or `auto` | v04.03: the JK-fit set matched to `%basis` (`aug-cc-pvtz-jkfit`; per element `def2-universal-jkfit` where PySCF has none); up to v04.02: `def2-universal-jkfit` | **Mean-field** (SCF, JK) fitting set, DF stages only. Must be a JK-fit set: `def2-universal-jkfit` (every element) or `aug-cc-pvtz-jkfit` (matched to aug-cc-pvtz; missing elements get def2-universal-jkfit). **Never an RI set** (section 7.1): until v04.01 this project used `aug-cc-pvtz-ri` here, which puts the SCF energy far too low; v04.02 moves an RI set given here to the correlation role |

### 5.3 Positrons

| Directive | Aliases | Value | Default | Notes |
|---|---|---|---|---|
| `%n_pos` | | 1, 2, ... | 1 | number of positrons |
| `%pos_basis` | `%positron_basis` | `PSX-DZ`/`PSX-TZ`/`PSX-QZ`, any basis name, `MATCH`, raw shells | `PSX-TZ` | section 8 |
| `%pos_mult` | | 1 or 3 | 1 | two positrons: 1 = singlet pair, 3 = triplet pair |
| `%pos_spin` | | 0 or 2 | 0 | same as `%pos_mult` - 1 |
| `%pos_aux` | | recipe | `etb` | one-positron DF auxiliary recipe |
| `%auxbasis_e` | | basis name or `default` | library default (`<electron basis>-ri`, e.g. `aug-cc-pvtz-ri`) | one-positron DF: electronic **correlation** fitting set, where RI sets belong. twoposBrorsen set it equal to its JK-fit set; `%auxbasis_e def2-universal-jkfit` reproduces its numbers (e.g. its LiH+e+ test to 1e-10 Ha) |
| `%df_aux` | | Python dict | `{}`; `auxbasis_e` per element: RI set for H-Ne, JK-fit set for Na and heavier (v04.04) | two-positron DF recipe (section 7.1); an RI `auxbasis_e` is replaced for Na and heavier, with a warning |
| `%pos_mass` | `%particle`, `%particle_mass` | `positron`, `antimuon` (`mu+`), `proton`, `deuteron`, `triton`, `muonic_helium`, or a mass (electron masses; `amu` suffix for amu) | `positron` | mass of the positive particle (v04.04, section 7.4). A negative muon is an error |
| `%quantum_site` | | atom number (1-based) of an H | none | the particle replaces that H nucleus (section 7.4); needs `%n_pos 1` |
| `%compute_eee` | | true/false | true | (T): pure-electron triples |
| `%compute_t2e1p` | | true/false | true | (T): electron-positron triples, where the CC class supports it |
| `%block_size_a` | | integer | library default | (T) block size, passed only if the (T) routine accepts it |

Booleans accept `true/false`, `yes/no`, `1/0`, `on/off` (any case).

### 5.4 Protons (NEO)

| Directive | Aliases | Value | Default |
|---|---|---|---|
| `%nuc_basis` | `%proton_basis` | nuclear basis | `pb4f1` |
| `%quantum_nuc` | | 0-based atom indices, comma-separated | `0` |
| `%nuc_mass` | | one value per `%quantum_nuc` entry, same order: a particle name, a mass in amu (`me` suffix for electron masses) or `default` | library table | v04.04, section 7.4; H nuclei only; needs pyscf_neo_cc 2026-09-15 or later |
| `%cneo` | | true/false | false | constrained NEO (v04.04, section 7.3); also the words `CNEO` / `CDFT` on the `!` line |
| `%cneo_variant` | | `ccd` / `ucd` | `ccd` | correlated CNEO only: where the constraint is imposed (section 7.3) |
| `%xc` | | functional | `b3lyp` |
| `%epc` | | `17-1`, `17-2`, `18-1`, `18-2` (`17`, `18`, `epc17` accepted), `none` | `17-2` |

### 5.5 Optimization

| Directive | Aliases | Value | Default | Notes |
|---|---|---|---|---|
| `%opt_engine` | `%optimizer` | `AUTO`, `INTRINSIC` (`GEOMETRIC`), `WOPT` (`WRAPPER`, `BFGS`), `ZOPT` (`ZMAT`, v04.05) | `AUTO` | section 10 |
| `%opt_max_cycles` | `%max_opt_cycles`, `%max_steps` | integer | 50 | both engines |
| `%conv_check` | | true/false | false | after OPT/ZOPT, also compute the exact-integral (CONV) energy at the final geometry. It forms exact four-index integrals (incl. the electron-positron tensor): the most memory-hungry step of a job. From v04.04 the final geometry, DF energy and summary are written first, so an out-of-memory kill during the check loses only the CONV number (and any later stages) |
| `%molden` | | true/false | false | v04.05: Molden files at the stage's final geometry: HF orbitals and CCSD natural orbitals, electrons and positrons in separate files (section 13) |
| `%cube` | | true/false | false | v04.05: cube files of the HF and CCSD densities, electrons and positrons (section 13) |
| `%cube_points` | | 10-400 | 80 | cube grid points per axis |
| `%chelpg` | | true/false | false | v04.05: CHELPG atomic charges at the final geometry, HF and CCSD, printed in the `.out` (section 13) |
| `%density_map` (one line per map) | | `<what> [lo,hi] [lo,hi] [colours] [ystep=<bohr>]`; what = electron / positron / both / esp / true / false | false | v04.05: plane maps of the HF and CCSD densities, or (`esp`) of the electrostatic potential of the nuclei and electrons, SVG (and PNG) in the job folder; `true` = electron (ELECTRONIC) or both (POSITRON); optional colour-scale limits (a.u., `auto` = default; with `both` electrons then positrons) and Multiwfn colour scheme (0-19 or its name); `ystep=` spacing of the vertical-axis labels in bohr (section 13) |
| `%ts_mode` | `%ts_type`, `%ts_search` | ts (`prfo`), irc, neb (`ci_neb`) | ts | v04.05 (2026-09-30): what a `! TS` / `! OptTS` stage runs (section 9.2) |
| `%ts_hessian` | | auto / first / each / never / stride:N / file:path | auto | v04.05 TS, IRC: starting Hessian (section 9.2) |
| `%ts_freq` | | true/false | true | v04.05 TS: frequency check at the result |
| `%irc_direction` | | both / forward / backward | both | v04.05 IRC |
| `%neb_product` | block | xyz lines, `end` | | v04.05 NEB: product geometry, same atoms in the same order as the stage geometry |
| `%neb_images` | | 5-50 | 11 | v04.05 NEB: number of images |
| `%neb_optimize_ends` | | true/false | false | v04.05 NEB: optimize the two end points too |
| `%freq_step` | | 1e-4 to 0.05 (bohr) | 0.005 (gradients), 0.01 (energies) | v04.05 FREQ: finite-difference step (section 9.1) |
| `%density_map_plane` | | AUTO / XY / XZ / YZ | AUTO | the plane of the maps; AUTO = the plane with the largest spread of the atoms, through their mean position |
| `%density_map_combine` | | `<job>[:electron\|positron] <job> ... [level=hf\|ccsd] [scale=shared\|own]` | false | v04.07: one figure with the maps of the named stages (this one or earlier) side by side, in the order given; species default positron (POSITRON) / electron (ELECTRONIC); shared colour scale per species and the first stage's plane by default (section 13) |
| `%constraints` | | block | | section 11 |
| `%pin` | `%pinning` | atom list `1,2` | | hold whole atoms fixed |
| `%pinnings` | | block `atom axis value` | | fix single Cartesian coordinates |
| `%equivalences` | | block `target source sx sy sz` | | coordinate equivalences |
| `%equivalence` | `%equiv` | `a1,a2,b1,b2` | | bond a1-a2 held exactly equal to bond b1-b2 (WOPT engine) |
| `%zopt_max_evals` | | integer | 60 | ZOPT (v04.01): maximum energies computed in this run |
| `%zopt_step_bond` / `%zopt_step_angle` | | Angstrom / degrees | 0.02 / 2.0 | ZOPT: first trial step of bond / angle and dihedral variables |
| `%zopt_tol_bond` / `%zopt_tol_angle` | | Angstrom / degrees | 0.001 / 0.1 | ZOPT: converged when the minimum is located this closely |
| `%zopt_etol` | | Hartree | 1e-6 | ZOPT: converged when less than this remains to gain |
| `%zopt_restart` | | true/false | true | ZOPT: reuse the energies in `<job>_zopt_history.json` |

### 5.6 Scans

| Directive | Value | Notes |
|---|---|---|
| `%scan r_min r_max n` | one line | fragment-gap scan; needs `! SCAN` |
| `%scan_fit` (`%fit`) | `morse`, `harmonic` (`harm`), `none` | v04.05: fit of the scan points (section 12); default none; ignored with a warning outside SCAN stages |
| `%scan` ... `end` | block, one scan line | `bond`, `translate`, `sym_zmat`, `fragment`, or `i j start end steps` (section 12) |

## 6. Geometry input

Coordinates are in Angstrom unless the stage says `! Bohrs` or `%coords bohr` (section 6.4).

### 6.1 Cartesian

```
* xyz <charge> <mult>
Cl   0.0   0.0  -1.9
Cl   0.0   0.0   1.9
*
```

- Closed by `*` or `end` (or by the next directive / end of stage, with no warning).
- Charge and multiplicity in the header are optional (`* xyz` alone keeps `%charge`/`%mult`).
- Symbols are case-normalized (`li` becomes `Li`, `CL` becomes `Cl`).
- If `%charge`/`%mult` and the header disagree, the value written **last** wins and a warning
  names both values.

### 6.2 Z-matrix

```
* zmat 0 1
O
H   1  roh
H   1  roh  2  aoh
C   1  1.43 2  109.5  3  120.0
roh = 0.96
aoh = 104.5
*
```

Row `i` is `label [ref r [ref angle [ref dihedral]]]`:

- References are earlier **row numbers (1-based)** or earlier **labels** (`O1`, `H2`).
- Values are numbers, variable names, or `-name`.
- Variables are defined inside the block as `name = value`, or scanned with
  `%scan sym_zmat` (the scanned variable needs no value). An undefined variable is an error.
  twoposBrorsen silently set it to 0.0, collapsing the atom.
- Labels may carry digits (`C1`, `H7`); the element is the label without digits.
- **Dummy atoms (v04.01):** a row labelled `X`, `XX`, `Du`, `Dum` or `Dummy` (digits allowed,
  `X1`) is placed like any atom and can be referenced, but is not part of the molecule: it has no
  nucleus, no electrons and no basis functions (electron or positron), and is removed before the
  calculation. All
  atom numbers elsewhere in the input (`%constraints`, `%pin`, `%scan bond`...)
  count real atoms only. `Xe` is xenon and `X-H` is a ghost label, not a dummy. A dummy on a
  symmetry axis lets a Z-matrix express the symmetry with fewer variables (NH3 below: two
  variables, exactly C3v).

```
* zmat 0 1
X
N  1  1.0
H  2  r  1  t
H  2  r  1  t  3   120.0
H  2  r  1  t  3  -120.0
r = 1.012
t = 111.8419       # X-N-H angle; H-N-H = 107.0 deg
*
```

Placement: atom 1 at the origin; atom 2 on +z from its reference; atom 3 in the xz-plane at
positive x; atoms 4+ by the standard NeRF construction. The angle is (new, ref, ref2). The
dihedral is (new, ref, ref2, ref3) with the IUPAC/Gaussian sign, which geomeTRIC also uses.
The test suite checks bond lengths, angles and dihedrals exactly (section 18).

> twoposBrorsen 00.81/00.86 placed the third atom at 180 deg minus the requested angle (its
> water tests came out at 75.5 deg instead of 104.5 deg, and its Li3H3 "equilateral" H3
> template was not equilateral). v04.00 places it correctly, so twopos Z-matrix inputs give
> the intended geometry here and a different one there.

A `* zmat` block whose lines are all `symbol x y z` is read as Cartesian, with a warning
(twoposBrorsen 00.73 test inputs do this).

### 6.3 Ghost atoms

A ghost atom carries basis functions but no nucleus and no electrons (counterpoise corrections, or a
basis-only site for a positron). Any PySCF spelling works in both `* xyz` and `* zmat` blocks:
`ghost-H`, `Ghost-H`, `ghost_H`, `ghostH` or `X-H` (element after the prefix). They are not dummy atoms:
`X`, `XX`, `Du` (section 6.2) are removed before the calculation, `X-H` is kept as a ghost.

- The charge and multiplicity in the header are those of the real atoms (a ghost adds no electrons).
- The electron basis goes on the ghost as on a real atom of that element; in the POSITRON sector the
  positron basis does too, unless the atoms on the `%pos_basis` line (v04.08) leave the ghost
  out. An element symbol there (`H`) also selects the ghosts of that element; use atom numbers to select only
  real atoms or only ghosts, for `%basis` as well (section 8.2).
- `%frozen auto` counts no core for a ghost (v04.04; up to v04.03 a ghost Cl froze 5 extra orbitals).
- Optimizers (geomeTRIC, WOPT, ZOPT) move ghosts like real atoms: the basis functions feel a force, and an
  unpinned ghost slides into the molecule (measured: ghost-H ended 0.44 A from F in HF + ghost-H, below the
  0.5 A close-atom limit, so the next stage fails validation). **Pin ghosts in optimizations** (`%pin <n>`)
  or, with ZOPT, write their Z-matrix values as numbers.
- The ghost label is kept in every output geometry and passed to the next stage (v04.04 also keeps the
  NEO isotope marks `H+` / `H*` after a geomeTRIC optimization; up to v04.03 they came back as plain H,
  so the final energy and the next stage used a proton).
- Two atoms closer than 0.5 A are an error for ghosts too (a ghost on top of a real atom only duplicates
  its basis functions).

Tested (`tests/test_ghost_atoms.py`, and `results/ghost_check_2026-09-24/`): every spelling, both geometry
forms, all three sectors, single point, gradient, geomeTRIC, WOPT and ZOPT.

### 6.4 Coordinate units: `! Bohrs` / `%coords` (v04.05)

Geometry blocks are read in Angstrom. To write them in Bohr, put `Bohrs` on the `!` line or `%coords bohr` in the
stage (ORCA's words; `! Angs` / `%coords angstrom` state the default explicitly):

```
! SP_ENERGY HF ELECTRONIC Bohrs
* xyz 0 1
H 0.0 0.0 0.0
H 0.0 0.0 1.3984
*
```

- What is converted: the Cartesian coordinates of `* xyz`, the bond lengths of `* zmat` (numbers and bond
  variables; angles and dihedrals stay in degrees) and the `%neb_product` coordinates. The wrapper converts
  them once, when it reads the stage, and works in Angstrom from then on: every output file and printed
  geometry is in Angstrom, and the stage summary says `written in Bohr, converted to Angstrom`.
- What is not: `%scan` ranges, `%constraints` targets and the ZOPT step sizes stay in Angstrom (a warning
  says so when a Bohr stage has them).
- The units belong to the stage that writes the coordinates. A stage without a geometry block inherits the
  previous geometry as it is (already converted); `Bohrs` there only gives a warning.
- `! Bohrs` together with a different `%coords`: warning, the one written last is used. An unknown unit is an
  error.
- Tests and examples: `tests/test_units.py`, `tests/inputs/units/`.

## 7. Sectors

### 7.1 POSITRON (`POS`)

The mean field is `POSHF` (electrons + positrons). If it does not converge with the default
solver, v04.00 retries with pyscf_neo_cc's ARH and Newton solvers, starting from the current
orbitals. If it still does not converge, a `[WARNING] ... NOT converged` line is printed
(twoposBrorsen marked such runs converged).

| Positrons | `DF` | `CONV` |
|---|---|---|
| 1 | `single_positron_dfccsd` (mean field: `%jkfit_aux_e`; correlation: `%auxbasis_e`, `%pos_aux`) | `single_positron_ccsd` |
| 2+, singlet pair | `apply_df_reference` + `POSDFCCSD(df_aux=...)` | `POSCCSD` |
| 2+, `%pos_mult 3` | `POSSODFCCSD(df_aux=...)` (spin-orbital) | `POSSOCCSD` |

**`%df_aux`** (two positrons, DF). A Python dict of pyscf_neo_cc DF options; keys
`auxbasis_e` (electronic correlation aux basis), `n_recipe` (positron aux recipe, default
`aug_etb`), `beta_n` (even-tempered ratio, default 2.0), `pos_label`. The default `{}` means
library defaults and is what v03.17 used; from v04.04 the wrapper fills `auxbasis_e` with the
JK-fit set matched to `%basis` when it is not given (see the fitting-set table below). Examples:

```
%df_aux {}                                              # defaults (auxbasis_e: matched JK-fit set)
%df_aux {'auxbasis_e': 'aug-cc-pvtz-jkfit', 'n_recipe': 'aug_etb'}
%df_aux aug-cc-pvtz-jkfit                               # shorthand for {'auxbasis_e': 'aug-cc-pvtz-jkfit'}
```

There is no memory key: memory is controlled by `%df_mem_cap`. In twoposBrorsen, leaving out
`%df_aux` put the library in its "exact-RI validation mode" (full rank-4 integrals, results
labelled DF). v04.00 always passes a dict, so `DF` always means real density fitting.

**Fitting sets and their roles (v04.02).** Density fitting uses two different kinds of
auxiliary set for two jobs:

| Job | Directive | Right kind of set | Default |
|---|---|---|---|
| mean field (SCF: Coulomb J and exchange K) | `%jkfit_aux_e` | JK-fit: `aug-cc-pvtz-jkfit`, `def2-universal-jkfit` | v04.03+: the JK-fit set matched to `%basis` (`aug-cc-pvtz-jkfit`); v04.02: `def2-universal-jkfit` |
| correlation, one positron | `%auxbasis_e` | RI: `aug-cc-pvtz-ri`, `cc-pvtz-ri` | library: `<electron basis>-ri` |
| correlation **and the pair CC's fitted Fock**, two positrons | `%df_aux {'auxbasis_e': ...}` | **per element**: RI for H-Ne, JK-fit for Na and heavier (an RI set given for heavy atoms is replaced) | v04.04: per element as described; v04.03: the library default (`<basis>-ri`, wrong for Cl and Br) |

An RI set has no tight functions for the core electrons. In the mean-field role a
Coulomb-metric fit then underestimates the electron repulsion and the SCF energy lands far too
low: for Br2 + e+ at aug-cc-pVTZ, -5150.06 Ha against -5145.18 Ha with exact integrals
(def2-universal-jkfit: within 4e-4 Ha; figures from the pyscf_neo_cc author, 2026-09-23). Every
CCSD, (T), gradient and geometry built on that mean field inherits the error.
pyscf_neo_cc 2026-09-23 refuses an RI set in the mean-field role (a name ending in `ri`). v04.02
therefore reads `%jkfit_aux_e <RI set>`, as written in most of this project's inputs, as the
correlation set it was meant to be: the RI set goes to `%auxbasis_e` (one positron) unless given,
the mean field gets the matched JK-fit set (v04.03), and the validation summary warns.

**Two positrons: `%df_aux`'s `auxbasis_e` is also a mean-field set (v04.04, 2026-09-25).** In the
two-positron path pyscf_neo_cc builds the CC's fitted reference (`apply_df_reference`) from the same
set it uses for the correlation, so that set must fit the core density too. With an RI set there
(the library's default `<basis>-ri`, which v04.02-v04.03 used, and where v04.02-v04.03 also moved an
RI `%jkfit_aux_e`) the two-positron energy is wrong although the printed POSHF energy is right:

| 2e+ system, CCSD(T) or CCSD | RI set | matched JK-fit set | measured by |
|---|---|---|---|
| Br2, aug-cc-pVTZ, CCSD(T) | 4.9 Ha too low | 0.9 mHa from exact | benchmark agent, `results/DF_error_2026-09-24/` |
| Cl2, aug-cc-pVTZ, CCSD(T) | 4.9 mHa above exact | 1.1 mHa below | same |
| F2, aug-cc-pVTZ, CCSD(T) | 0.24 mHa below exact | 1.1 mHa below | same |
| Cl2, cc-pVDZ, CCSD | 12.0 mHa below exact | 0.60 mHa | `results/twopos_df_aux_check_2026-09-25/` |
| F2, cc-pVDZ, CCSD | 3.2 mHa below exact | 0.08 mHa | same |

One positron is not affected (its reference is the SCF itself: Br2 + e+ with the RI correlation set
0.63 mHa from exact; Cl- + e+ MP2 changes by 0.01 mHa between the two sets).

**DZ JK-fit sets and Ga-Kr (v04.05).** PySCF's `cc-pvdz-jkfit` and `aug-cc-pvdz-jkfit` entries for Ga-Kr
have no g functions and cannot fit the filled 3d shell. RHF, fitted minus exact, at cc-pVDZ / aug-cc-pVDZ:

| Ga+ | Ge2+ | As3+ | Se2- | Br- | Kr | Cl- (for comparison) |
|---|---|---|---|---|---|---|
| +482 mHa | +534 | +585 | +627 | +674 | +719 | 0.00 |

v04.05 uses `aug-cc-pvtz-jkfit` for those elements instead (Br-: 0.002 / 0.004 mHa), in the mean field and in
the two-positron pair set, and prints a warning. This applies whether the DZ set was the default (matched to a
DZ `%basis`) or written in the input. v04.04 only warned. Zn has no DZ JK-fit set in PySCF and gets the
fallback.

**The two-positron standard (v04.04, 2026-09-26): per element.** `%df_aux`'s `auxbasis_e` defaults to the RI
set matched to `%basis` for H-Ne (the more accurate set there: F2 0.24 against 1.07 mHa) and the JK-fit set
of the mean field for Na and heavier (the only correct kind there). A molecule with both gets an
element-keyed set: H-Cl(2-) + 2e+ at cc-pVDZ, CCSD, is 0.19 mHa from exact with it (all JK-fit 0.38, all
RI 5.5 mHa). Where the basis has no RI set (6-31G) the light elements get the JK-fit fallback
(def2-universal-jkfit). An explicit RI set is kept for H-Ne and replaced for heavier elements (warning);
any other explicit set is used as given. The RI set also keeps C/N gradients possible: see the pre-flight
check below. The same set is used by the two-positron MP2 (`POSDFMP2`) and the triplet pair. No RI
`%jkfit_aux_e` is moved into `%df_aux` any more.

**Pre-flight check of the two-positron DF gradient (v04.04).** The analytic DF gradient needs a Cholesky
factor of every fitting metric. With a nearly linearly dependent fitting set the library falls back to an
eigen-decomposition; the energy still runs, but the gradient stops, after the full CC: (CN)2(2-) + 2e+ with
`aug-cc-pvtz-jkfit` as the pair set (aug-cc-pVTZ, PSX-TZ) makes the unified electron-positron fit 801 x 836
and stopped after 2 h 07 min on the cluster. The factors exist right after the SCF, so every two-positron
DF stage that will need the gradient (OPT, GRADIENT, ZOPT with gradients; CCSD, CCSD(T), MP2) checks them
there and, if one is not a Cholesky factor, stops with exit code 3 and says which set to use (18 s for
that case); otherwise it prints `[Pre-flight] ... every fitting metric has a Cholesky factor`. Measured with
the same bases: (CN)2 is fine with `aug-cc-pvtz-ri` or `def2-universal-jkfit`; Br2 is fine with all three
sets. SP_ENERGY and SCAN are not checked (the energy tolerates the fallback). The check needs pyscf_neo_cc 2026-09-23 (it reads the fitting factors the POSHF builds); with the older library it does not run and prints nothing. Script to check a system in
seconds: `results/Tests/twopos_df_aux_check_2026-09-25/gradient_metric_check/metric_check.py`. **Two-positron DF energies from v04.02-v04.03 or v03.17 with
Cl, Br (or any RI `auxbasis_e`) carry this error**; fragments with one positron do not, so
two-positron channels against one-positron fragments do not cancel it (Cl2: 2e+ Cl2 -> 2 PsCl is
-10.38 kcal/mol with exact integrals, -6.94 with the v04.03 default, -10.41 in the old all-RI setup).

**DF recipe on the mean field (v04.02, pyscf_neo_cc 2026-09-23).** On DF stages the mean
field receives the same DF recipe the CC uses (`%auxbasis_e`/`%pos_aux` for one positron,
`%df_aux` for two), so the SCF runs once on the fitted electron-positron Coulomb and the exact
four-index electron-positron tensor (43 GB for a six-centre positron basis) is never formed; the
CC reuses the fits. The old route (exact SCF, then re-converged on the fit) gives the same
correlated energies (tested to 1e-7 Ha; 1.0e-9 Ha for LiH+e+ CCSD). The printed `Converged POSHF
Energy` of a DF stage is now the SCF on the fitted electron-positron Coulomb, i.e. the reference
the CC actually uses; before, it was the exact-e-p SCF, which the CC then re-converged (the two
differ by ~1e-7 Ha: 2.5e-7 Ha for LiH+e+ at STO-3G). The wrapper detects the feature
(`POSHF(df_aux=...)`) and keeps the old route on the older library.

**Matched fitting sets (v04.03).** Measured at aug-cc-pVTZ, SCF error of each mean-field set
against exact integrals (results/fitting_set_decision_2026-09-23/REPORT.md):

| Species | `aug-cc-pvtz-ri` (until v04.01) | `def2-universal-jkfit` (v04.02) | `aug-cc-pvtz-jkfit` (v04.03) |
|---|---|---|---|
| H- | -0.057 mHa | -2.42 mHa | -0.054 mHa |
| PsH | -0.002 mHa | -0.32 mHa | -0.0004 mHa |
| CN- | -0.034 mHa | +0.055 mHa | +0.085 mHa |
| F- | +0.024 mHa | -0.002 mHa | +0.002 mHa |
| Cl- | +2.80 mHa | -0.25 mHa | -0.0004 mHa |
| Br- | -2441 mHa | -0.36 mHa | +0.012 mHa |
| HBr2- + e+ / Br2 2- + e+ | -2441 mHa / (author: -4884) | -0.80 / -0.43 mHa | +0.015 / +0.029 mHa |

So v04.03 uses, when `%jkfit_aux_e` is not given (or is `auto`), the JK-fit set matched to
`%basis` (`<basis>-jkfit`), element by element, with `def2-universal-jkfit` for elements PySCF has
no matched set for (e.g. Li with the cc sets) and for bases without matched sets (6-31G, STO-3G).
An RI set moved out of the mean field is replaced the same way. In the ELECTRONIC sector the
correlation of a DF-CC stage is fitted with `%auxbasis_e`, else the matched RI set
(`<basis>-ri`), else the mean-field set; up to v04.02 it used the mean-field set, which moved CN-
by -0.71 mHa. (Stock PySCF has no analytic DF-CCSD gradient, so gradients of electronic DF stages
are numerical and consistent with these energies.) The validation summary prints the sets each
stage uses ("Fitting sets: ...").

**CONV means an exact mean field (v04.02).** A `CONV` stage, and the `%conv_check`
cross-check, build `POSHF` without any fitting. Up to v04.01 the CONV mean field was still
JK-fitted (only the correlation was exact), and pyscf_neo_cc 2026-09-23 refuses canonical
gradients over a JK-fitted mean field. CONV energies therefore differ slightly from v04.01's
(the JK-fit error of the mean field; 0.4 mHa for PsH at 6-31G).

**Frozen core** (`%frozen`) is applied to every CC class. **Triples** (`CCSD(T)`) pass
`%compute_eee`, `%compute_t2e1p` and `%block_size_a` only to (T) routines that declare them.
The one-positron (T) accepts `compute_eee`; the two-positron (T) accepts neither flag.

### 7.2 ELECTRONIC (`ELEC`)

Plain PySCF: RHF, or UHF when `%mult` != 1 (RKS/UKS for `DFT` with `%xc`), `cc.CCSD`, and
`ccsd_t()`. With `DF` the mean field is density-fitted with `%jkfit_aux_e`, and CCSD follows it.

- **Gradients:**
  - Analytic for HF and DFT (also with DF), MP2 (CONV), CCSD (CONV, closed and open shell), and closed-shell
    CCSD(T) (CONV). CCSD(T) uses PySCF's CCSD(T) gradient with the (T) lambdas, checked against finite
    differences. The earlier wrappers used the CCSD gradient with the CCSD(T) energy, which is inconsistent.
  - Numerical for DF-CCSD and DF-CCSD(T): PySCF's CC gradient formula applied to DF amplitudes is not the
    derivative of the DF energy (2.6e-5 and 5.2e-5 Ha/Bohr off for CN- at aug-cc-pVTZ).
  - Numerical for DF-MP2 and open-shell CCSD(T), which have no PySCF gradient.
  - Up to v04.04 electronic CCSD gradients were silently numerical, because PySCF 2.14's CC classes have
    `nuc_grad_method()` but no `Gradients()`. And a `GRADIENT` job without an analytic gradient started an
    optimization; v04.05 returns a numerical gradient instead (the same fix applies to positron HF).
- **OPT:**
  - v04.05: geomeTRIC, through PySCF's driver, with the gradients above. It uses about a third of WOPT's
    steps on water (4 against 11-13) and reaches the same minimum.
  - `%opt_engine WOPT` still runs the built-in engine, which is also the fallback if geomeTRIC
    fails.
  - Up to v04.04 every electronic OPT ran WOPT.

### 7.3 PROTON (`NEO`) and constrained NEO (`CNEO`, `CDFT`)

The nuclei in `%quantum_nuc` (0-based) are quantum particles in `%nuc_basis`. Methods: HF, DFT (`%xc`,
`%epc`: 17-1, 17-2, 18-1, 18-2 or `none`), MP2 (section 7.5), CCSD and CCSD(T).

**Plain NEO vs constrained NEO (CNEO, v04.04).** In plain NEO the quantum nucleus is free inside its
basis, so a geometry optimization moves the *centre of the nuclear basis functions*, not the nucleus: its
"bond lengths" are not nuclear positions. CNEO adds a Lagrange multiplier that holds the expectation value
of each quantum nucleus's position at its geometry coordinate, so the optimized geometry is a real
geometry (Yang's constrained NEO; the correlated methods follow Tucker & Brorsen, JCP 164, 184117 (2026)).
Measured (HF molecule, 6-31G, pb4d, H quantum; `results/neo_dft_epc_gradients_2026-09-25/`), optimized H-F:

| Level | Classical H | Plain NEO | CNEO |
|---|---|---|---|
| HF | 0.9209 A | 0.8654 A | 0.9404 A |
| B3LYP (+ epc17-2 for the quantum H) | 0.9495 A | 0.9057 A | 0.9661 A |
| MP2 | 0.9470 A | 0.8967 A | 0.9662 A |
| CCSD(T) | | | 0.9677 A (`ccd`) |

CNEO lengthens the bond by about 0.02 A over a classical nucleus at every level (the proton's zero-point
motion); CNEO-MP2, CNEO-CCSD(T) and CNEO-B3LYP+epc agree within 0.002 A. **Use CNEO for geometries with
quantum nuclei.**

How to ask for it: `%cneo true` in a PROTON stage, or the word `CNEO` on the `!` line (`! OPT CCSD(T) CNEO
DF`), or `CDFT` for constrained NEO-DFT (`! OPT CDFT` = `DFT CNEO`; CDFT is the library's class name).

| Method | CNEO code (pyscf_neo_cc) | Integrals | Analytic gradient |
|---|---|---|---|
| HF | `CDFT(xc='HF')`, converged as `cneo.cneo_hf` (DIIS, then the ARH solver to 1e-12) | exact | yes |
| DFT | `CDFT(xc=%xc, epc=%epc)`; the epc term is in the gradient | exact | yes |
| MP2 | `CNEOMP2` / `CNEODFMP2` | CONV or DF | yes |
| CCSD, CCSD(T) | `CNEOCCSD` / `CNEODFCCSD`, `.ccsd_t()` | CONV or DF | yes, (T) included |

DF correlated CNEO uses a density-fitted reference (`cd_density_fit_unified`: mean field = the JK-fit set
matched to `%basis`, correlation = `%auxbasis_e` or the matched RI set), which its DF gradient requires.

**`%cneo_variant` (correlated CNEO only): `ccd` (default) or `ucd`.** `ccd` imposes the constraint on the
correlated density; `ucd` only on the HF reference, and then the correlation lets the nucleus drift:
optimized H-F with CNEO-CCSD 0.9667 (`ccd`) against 0.9001 A (`ucd`), with CNEO-CCSD(T) 0.9677 against
0.8957 A, i.e. `ucd` geometries fall back towards plain NEO's. The library's own default is `ucd`; the
wrapper uses `ccd` and warns when an optimization asks for `ucd`. For CNEO-CCSD(T) with `ccd` the
multiplier is found on the CCSD(T) density (`constraint_level='ccsd_t'`), which the exact (T) gradient
needs; it costs about 2.5x the CNEO-CCSD time. For CNEO-MP2 the variants differ by 3e-5 A.

**Gradients and optimizers (v04.04).** HF and DFT, plain or constrained, use the library's analytic NEO
gradient and geomeTRIC through PySCF's driver; correlated methods use their analytic gradients and
`pyscf_neo_cc.geomopt.optimize`. Up to v04.04 the wrapper called a gradient method NEO-HF / NEO-KS do not
have, so their GRADIENT and OPT jobs fell back to WOPT with numerical gradients (a GRADIENT job then ran an
optimization). Still numerical: plain NEO DF-CCSD / CCSD(T) (the library's DF gradient needs a
density-fitted mean field the plain-NEO path does not build yet). `%frozen` now applies in every PROTON
path (it was ignored by single points and geomeTRIC: 0.88 mHa on HF/6-31G). `%nuc_mass` and the isotope
symbols work with CNEO.

Checks (`tests/test_cneo.py`): energies equal the library called directly (1e-8 Ha: CNEO-HF, CDFT,
CNEO-DF-MP2, CNEO-CCSD(T)); gradients equal finite differences (2e-6 Ha/Bohr, typically 1e-8); geomeTRIC
optimizations reach the library-level distances above with no fallback.

### 7.4 Other particles: antimuons, protons, deuterons, tritons (v04.04)

pyscf_neo_cc treats two kinds of quantum particle besides the electrons, and both must be **positive**:

- the POSITRON sector's particle: mass `pos_mass` (electron masses), charge +1;
- the PROTON sector's quantum nuclei (NEO): hydrogen nuclei with the library's masses, or the
  isotope symbols `H+` (deuteron), `H*` (antimuon as a light H nucleus) and `H#` (muonic helium, a
  He-4 nucleus with its muon: charge +1, 4.115 amu), or since 2026-09-15 any mass list.

v04.04 gives both routes the same particle table (bare particles, the masses the NEO Mole uses):

| Name (aliases) | Mass (m_e) | Mass (amu) | POSITRON sector | PROTON sector |
|---|---|---|---|---|
| `positron` (`e+`) | 1 | 0.000549 | default | - |
| `antimuon` (`mu+`, `muon+`) | 206.768 | 0.11343 | `%pos_mass antimuon` | `H*`, or `%nuc_mass antimuon` |
| `proton` (`p`) | 1836.153 | 1.00728 | `%pos_mass proton` (library check) | default |
| `deuteron` (`d`) | 3670.48 | 2.01355 | `%pos_mass deuteron` | `H+`, or `%nuc_mass deuteron` |
| `triton` (`t`) | 5496.92 | 3.01550 | `%pos_mass triton` | `%nuc_mass triton` (no library symbol) |
| `muonic_helium` (`hemu`) | 7501.07 | 4.11494 | `%pos_mass muonic_helium` | `%nuc_mass muonic_helium` (the library's `H#`; `#` starts a comment in the input, so `H#` cannot be written) |

**Not possible:** a **negative muon** (or an antiproton) as a free particle: the library's particles
are positive, so `%pos_mass muon` is an error that says so. A negative muon exists here only inside
muonic helium. "Muon" in muonium chemistry means the positive antimuon: write `antimuon` or `mu+`.

**POSITRON route** (`%pos_mass`, `%quantum_site`): the particle replaces the nucleus of one H atom
(the library's "proton-limit" mode: that nuclear charge is removed, the electron basis stays, the
particle basis sits there only). The molecular charge and multiplicity are the ordinary ones (HF is
`* xyz 0 1`). Everything the POSITRON sector does works: HF, CCSD, CCSD(T), DF or CONV, gradients,
OPT/WOPT/ZOPT, scans. `%quantum_site` needs `%n_pos 1` and an H atom.

```
! OPT CCSD(T) POSITRON DF
%pos_mass antimuon
%quantum_site 1          # atom 1 (the H) becomes Mu+: MuF
%pos_basis pb4f1         # protonic set, scaled to the antimuon mass
%basis aug-cc-pvtz
* xyz 0 1
H 0 0 0
F 0 0 0.917
*
```

**PROTON route** (`%nuc_mass`): one value per `%quantum_nuc` entry, in the same order (a name, a
mass in amu, or `default`). H nuclei only; an atom written `H+`/`H*` already has its mass (combining
it with `%nuc_mass` is an error). Needs pyscf_neo_cc 2026-09-15 or later (the MCCC environment's
older library refuses it at validation; `H+`/`H*` and the POSITRON route work there). Methods: those
of the PROTON sector (NEO-HF/DFT, NEO-CCSD, CCSD(T) where available).

**Basis sets. No basis has to be imported.** pyscf_neo_cc ships protonic nuclear basis sets
(`pb4d`, `pb4f1`, `pb4f2`, `pb5d`, `pb5f`, `pb5g`, `pb6d`, `pb6f`, `pb6g`, `pb6h`, even-tempered
`8s8p8d8f`, `8s8p8d8f8g`), all made for the proton. For another mass the library's own rule (used for
`H+`, `H*`, `H#`) multiplies every exponent by sqrt(m / m_proton): a heavier particle is more
localized. The wrapper applies the same rule on both routes: `%pos_basis pb4d` is loaded from the
library and scaled to `%pos_mass`, and `%nuc_mass` scales `%nuc_basis` (the library itself would leave
a mass list in the proton's basis). Scale factors: antimuon 0.3356, proton 1, deuteron 1.4139,
triton 1.7302, muonic helium 2.0212. Positron sets (PSX, aug-cc) are far too diffuse for a heavy
particle; the wrapper warns. See the table below for how the energy converges with the set.

**Supported particle basis sets** (shipped with pyscf_neo_cc in `pyscf_neo_cc/_neo/basis/`; the same
names work in `%nuc_basis` (PROTON) and in `%pos_basis` (POSITRON, v04.04), and are scaled to the mass on
both routes; names are matched as the library does, so `PB4-D`, `pb4d` and `pb_4d` are the same set).
Functions are spherical (Cartesian in parentheses); exponents are for the proton, before scaling.

| Name | Shells | Functions | Exponents (proton) | Source / purpose |
|---|---|---|---|---|
| `pb4d` | 4s3p2d | 23 (25) | 1.96 - 32.0 | PB4-D, Yu, Pavosevic & Hammes-Schiffer, J. Chem. Phys. 152, 244123 (2020); the library's default for H |
| `pb4f1` | 4s3p2d1f | 30 (35) | 5.97 - 29.0 | PB4-F1 (same paper); **the wrapper's default `%nuc_basis`** |
| `pb4f2` | 4s3p2d2f | 37 (45) | 5.97 - 29.0 | PB4-F2 |
| `pb4f2a` | 4s3p2d2f | 37 (45) | 5.23 - 30.5 | PB4-F2a (variant of PB4-F2) |
| `pb5d` | 5s4p3d | 32 (35) | 1.91 - 40.1 | PB5-D |
| `pb5f` | 5s4p3d2f | 46 (55) | 2.35 - 48.5 | PB5-F |
| `pb5g` | 5s4p3d2f1g | 55 (70) | 3.17 - 47.0 | PB5-G |
| `pb6d` | 6s5p4d | 41 (45) | 2.17 - 38.8 | PB6-D |
| `pb6f` | 6s5p4d3f | 62 (75) | 1.42 - 39.8 | PB6-F |
| `pb6g` | 6s5p4d3f2g | 80 (105) | 2.02 - 29.3 | PB6-G |
| `pb6h` | 6s5p4d3f2g1h | 91 (126) | 0.51 - 36.9 | PB6-H (largest) |
| `8s8p8d8f` | 8s8p8d8f | 128 (160) | 2.83 - 32.0 | even-tempered, alpha = 2 sqrt 2, beta = sqrt 2 |
| `8s8p8d8f8g` | 8s8p8d8f8g | 200 (280) | 2.83 - 32.0 | even-tempered (the library's HeHHe+ tests) |
| `NsNpNd`, `NsNpNdNf` (any N, e.g. `6s6p6d`) | as named | | from 2.83 | even-tempered, built as the library builds it for H |
| `dzsnb` | 2s | 2 | 20.2, 29.3 | Webb, Iordanov & Hammes-Schiffer, J. Chem. Phys. 117, 4106 (2002); minimal, for comparisons only |
| `1s1p`, `2s1p` | 1s1p, 2s1p | 4, 5 | 20 - 29 | same paper, minimal |
| `sp_well_sep` | 1s1p | 4 | 4, 24 | library test set, not for results |
| `1e6` | 1s | 1 | 10^6 | a nearly classical (point-like) nucleus, for tests |

The PB sets are the ones meant for results; within a family, the f and higher functions matter for the
correlated methods (see the convergence table below). A heavy quantum nucleus other than H (PROTON
sector) gets the library's default even-tempered 12s12p12d set from its mass; `%nuc_mass` is for H only.
**When to import a basis:** only to use a set made for a specific particle (for example a published
muonic basis) instead of a scaled protonic one; give it as raw shells, `%pos_basis [[0,[exp,1.0]],...]`
(POSITRON route; raw shells are used as given, never scaled). The PROTON route cannot take raw shells
(the library reads its nuclear basis by name only).

**How good each set is** (measured 2026-09-24, `results/particle_mass_basis_2026-09-24/`): HF molecule
(F-H 0.917 A) with the particle in place of the H nucleus, POSITRON route, CONV, cc-pVTZ electrons. Energy
above the largest set, `pb6h`, in mHa: CCSD(T), with HF in parentheses.

| Basis | antimuon | proton | deuteron | triton |
|---|---|---|---|---|
| `pb4d` | +3.67 (+0.086) | +0.80 (+0.029) | +0.42 (+0.025) | +0.29 (+0.023) |
| `pb4f1` | +1.53 (+0.056) | +0.25 (+0.016) | +0.11 (+0.012) | +0.07 (+0.011) |
| `pb4f2` | +1.41 (+0.056) | +0.23 (+0.014) | +0.09 (+0.010) | +0.05 (+0.008) |
| `pb5d` | +3.65 (+0.081) | +0.79 (+0.008) | +0.42 (+0.005) | +0.29 (+0.004) |
| `pb5f` | +1.42 (+0.058) | +0.23 (+0.005) | +0.09 (+0.002) | +0.05 (+0.001) |
| `pb5g` | +0.30 (+0.015) | +0.04 (+0.002) | +0.02 (-0.000) | +0.01 (-0.001) |
| `pb6d` | +3.64 (+0.081) | +0.79 (+0.006) | +0.42 (+0.003) | +0.29 (+0.002) |
| `pb6f` | +1.34 (+0.053) | +0.22 (+0.003) | +0.09 (-0.001) | +0.05 (-0.002) |
| `pb6g` | +0.19 (+0.011) | +0.04 (-0.003) | +0.02 (-0.004) | +0.01 (-0.004) |
| `pb6h` | +0.00 (+0.000) | +0.00 (+0.000) | +0.00 (+0.000) | +0.00 (+0.000) |
| `8s8p8d8f` | +1.33 (+0.051) | +0.21 (+0.002) | +0.08 (-0.001) | +0.05 (-0.003) |
| `8s8p8d8f8g` | +0.11 (-0.017) | +0.03 (-0.005) | +0.01 (-0.005) | +0.00 (-0.005) |
| PSX-TZ (positron basis) | +55.91 (+39.736) | +78.56 (+85.700) | +83.81 (+95.598) | +86.45 (+100.273) |
| cc-pVTZ (electron basis) | +34.88 (+34.817) | +56.30 (+84.436) | +61.52 (+94.599) | +64.14 (+99.363) |

Reading it: the highest angular momentum in the set matters much more than the number of s/p functions (d-only
sets are alike, so are the f sets and the g sets), and a lighter particle needs more of it (antimuon: 3.6 mHa
with d functions only, 1.4 with f, 0.2-0.3 with g; proton: 0.8, 0.23, 0.04). HF is converged by any PB set
(< 0.1 mHa). A positron basis or the electron basis at the site is 35-100 mHa off: do not use them for heavy
particles. `pb6h` itself is not the basis-set limit; relative energies (the same particle and set on both
sides of a reaction) usually converge faster than these totals (not measured here). Cost at this size (on a busy machine): `pb4f1` 15-45 s, `pb6h` 35-85 s,
`8s8p8d8f8g` 3-4 min per CCSD(T) point.

**Checks** (`tests/test_masses.py`): the particle masses equal the library's own H/H+/H*/H# masses
and the scaled sets equal its H+/H*/H# sets to 1e-14; the POSITRON route at the proton mass
reproduces NEO (HF 1e-9 Ha, CCSD and CCSD(T) 1e-7 Ha); deuteron and antimuon agree on three routes
(POSITRON sector, library symbol, `%nuc_mass`), the triton and muonic helium on two; energies are
ordered by mass (lighter particle = more zero-point energy); DF - CONV is 0.06 mHa for the antimuon
(6-31G); geomeTRIC optimizations on the two routes give the same bond to 2e-4 A.

### 7.5 MP2 (v04.04)

`! ... MP2 ...` in any sector. Each sector uses the library route that has one:

| Sector | Integrals | Code | Gradient | Notes |
|---|---|---|---|---|
| ELECTRONIC | CONV | PySCF `MP2` (`UMP2` for open shells) | analytic | |
| ELECTRONIC | DF | PySCF `DFMP2` / `DFUMP2`, correlation fitted with the RI set matched to `%basis` (or `%auxbasis_e`) | **numerical** (PySCF has no DF-MP2 gradient): 6 energies per atom per step, with a warning | use CONV for optimizations |
| POSITRON, 1 e+ | DF only | pyscf_neo_cc `DFMP2` on the POSHF built with the stage's DF recipe (`%auxbasis_e`, `%pos_aux`) | analytic | |
| POSITRON, 2 e+ | DF only | `POSDFMP2`, closed-shell pair, `%df_aux` recipe | analytic | `%pos_mult 3` and odd `%n_pos` are errors |
| PROTON | DF only (CNEO: CONV or DF) | `DFMP2` on a NEO-HF set up with `cd_density_fit_unified` (mean field: matched JK-fit set; correlation: matched RI set); CNEO: `CNEOMP2` / `CNEODFMP2` (section 7.3) | analytic | |

- `%frozen` applies in every sector. The summary and `.out` report the reference (HF) energy, the MP2
  total and the correlation energy.
- There is no conventional (exact-integral) positron or NEO MP2 in pyscf_neo_cc: `CONV` there is an error,
  and `%conv_check` is skipped with a warning.
- All engines work: geomeTRIC (POSITRON, PROTON), WOPT, ZOPT, scans. A `GRADIENT` job gives the gradient
  (numerical where no analytic one exists) and never turns into an optimization.
- Needs pyscf_neo_cc 2026-09-23 for the POSITRON and PROTON sectors (validation says so on older ones).
- Checked (`tests/test_mp2.py`, `results/lih2_hf_mp2_2026-09-24/`): every route equals a direct PySCF /
  pyscf_neo_cc calculation to 1e-8 Ha, analytic gradients equal finite differences to 2e-6 Ha/Bohr, and on
  (LiH)2 geomeTRIC, WOPT and ZOPT reach the same MP2 energy (to 2e-8 Ha) in every route that converges.
- **Known limit:** e+ (LiH)2 with one positron. POSHF does not converge at geometries that break the D2h
  symmetry by as little as 0.002 A (max_cycle 3000 does not help), and optimizers make such geometries
  (finite-difference displacements, numerical noise). MP2 then stops with "mean field not converged".
  ZOPT keeps the symmetry exactly and works (E = -16.0401126 Ha).

## 8. Positron basis sets and atom selection

### 8.1 PSX-DZ, PSX-TZ, PSX-QZ

The PSX positron sets are the openLOWDIN ones (`openLOWDIN/lib/basis/PSX-DZ`, `PSX-TZ`,
`PSX-QZ`; the same exponents for every element), copied into the wrapper so it also runs where
openLOWDIN is not installed. A test checks every exponent against those files.

| Name | Set | Shells | Functions per atom (spherical) |
|---|---|---|---|
| `PSX-DZ` | 5s3p2d | 10 | 24 |
| `PSX-TZ` | 6s4p3d2f | 15 | 47 |
| `PSX-QZ` | 7s5p4d3f2g | 21 | 81 |
| `PSX-5Z` (2026-10-06) | 8s6p5d4f3g2h | 28 | 128 |
| `PSX-6Z` (2026-10-15, v04.08) | 9s7p6d5f4g3h2i | 36 | 190 |

`PSX-5Z` is not an openLOWDIN set: it was built in this project with the recipe of the other three (Moncada
et al., Chem. Sci. 11, 44 (2020), ESI) and continues them smoothly; parameters and tests in
`results/Tests/PSX_5Z_2026-10-06/REPORT.md`. `PSX-6Z` (v04.08) extends it with the same recipe to i
functions, for aug-cc-pV6Z; build and benchmark in `results/Tests/PSX_6Z_2026-10-15/` (Li2 correlation is
still not basis-converged at 6Z).

All uncontracted. The stage summary prints the set used (e.g. `PSX-TZ = 6s4p3d2f (15 shells)`).
Functions are spherical (pyscf_neo_cc's `build_system` default `cart=False`, used for electrons and positrons). openLOWDIN
uses Cartesian functions, so there the same sets have 26 / 56 / 106 functions per atom and its
energies will not match ours exactly; measured at CCSD(T)/6-31G the difference is only 0.006-0.039 mHa
(<= 0.013 kcal/mol in bond energies). The published (Basis Set Exchange) PsX exponents differ from
openLOWDIN's by <= 0.3% (rounding) and change energies by < 0.001 mHa (`results/PSX_impact_631G/REPORT.md`,
sections 1.1, 7, 8).

**Earlier wrappers did not use exactly these sets:**

| Wrapper | Its `PSX-DZ` | Its `PSX-TZ` |
|---|---|---|
| newBRORSEN v03.17 (`_localfix`/`_merged`) | a different 2s1p1d set (4 shells, 10 functions) | missing the tightest s (exponent 1.8214) |
| twoposBrorsen, Brorsen-MCCCSDT-v5 (`old_wrapper`), pyscf_neo_cc's own test | missing the tightest s (exponent 1.0604) | complete |

Effect on e+.[H- Cl-] (R = 3.921 A), aug-cc-pvdz electrons, aug-cc-pvtz-ri JK fit, frozen
core, DF-CCSD (openLOWDIN set minus the earlier set):

| Earlier set | CCSD energy difference |
|---|---|
| v03.17 `PSX-DZ` (2s1p1d) | -16.4 mHa (-10.3 kcal/mol) |
| twopos / v5 `PSX-DZ` (4s3p2d) | -0.59 mHa (-0.37 kcal/mol) |
| v03.17 `PSX-TZ` (5s4p3d2f) | -0.23 mHa (-0.15 kcal/mol) |

So results from different wrappers used slightly different positron bases, and the v03.17
DZ results a much smaller one. A full test at CCSD(T)/6-31G on six positronic dianions, one and two
positrons (`results/PSX_impact_631G/REPORT.md`, 2026-09-23), found: v03.17's PSX-TZ changes
bond energies by at most 0.044 kcal/mol and distances by at most 0.003 A (negligible);
v03.17's PSX-DZ by up to 8 kcal/mol and 0.41 A (unusable); the twopos/v5 PSX-DZ by up to
0.47 kcal/mol and 0.018 A. Do not mix them in one comparison. To reproduce an old number,
give that wrapper's exponents explicitly as raw shells (`%pos_basis [[0,[...,1.0]], ...]`).

### 8.2 `%pos_basis` forms

| Form | Meaning |
|---|---|
| `%pos_basis PSX-TZ` | the set on every positron-carrying atom |
| `%pos_basis PSX-TZ H` / `PSX-TZ {N O}` (v04.08) | the set on every atom of those elements only; the others get no positron functions |
| `%pos_basis PSX-TZ 1 2` / `PSX-TZ {1 2}` (v04.08) | the set on atoms 1 and 2 only (1-based real atoms) |
| `%pos_basis PSX-DZ H PSX-TZ N` (v04.08) | PSX-DZ on the H atoms, PSX-TZ on the N atoms |
| `%pos_basis PSX-DZ {1} PSX-TZ` (v04.08) | PSX-DZ on atom 1, PSX-TZ (no atoms after it) on all the others |
| `%pos_basis PSX-TZ [1,5] PSX-DZ` (v04.09) | PSX-TZ on atoms 1 to 5 (inclusive; `[1-5]` also works), PSX-DZ on all the others. Same on `%basis` |
| `%pos_basis MATCH` | the electron basis (also `E_BASIS`) |
| `%pos_basis aug-cc-pvdz` | any PySCF basis name |
| `%pos_basis NONE` | no explicit positron basis (library default: the electron basis) |
| `%pos_basis [[0,[0.1,1.0]], [1,[0.2,1.0]]]` | raw PySCF shells `[l, [exponent, coefficient]]` |

**Atoms on the basis line (v04.08).** `%pos_basis` and `%basis` take the atoms each basis goes on, so
several bases can share one line. After a basis name come element symbols
and/or atom numbers (braces and commas only group: `{1 2}` = `1 2`); the next basis name starts a new group.
An atom number beats an element symbol, which beats a basis written with no atoms (the default for the rest):
`%pos_basis PSX-DZ H PSX-TZ {3}` puts PSX-TZ on atom 3 even if it is an H. Two different bases for the same
atom at the same level (two element groups, or two atom numbers) are an error, as are an element not in the
geometry and an atom number past the last atom.

- Ghost atoms (`ghost-H`, `X-H`) count as atoms: an element symbol also selects that element's ghosts, a basis with no atoms after it covers them, and atom numbers count them in geometry order (dummy atoms `X` of a Z-matrix are not counted). To give a ghost a different basis from the real atoms of its element, select it by number; `ghost-H` itself cannot be written on the basis line (it would be read as a basis name). Checked 2026-10-14 (dry run, LiH + ghost-H at atom 3):
  `%pos_basis PSX-TZ H` / `%basis aug-cc-pvdz H cc-pvdz` put the H sets on H2 and ghost-H3;
  `%pos_basis PSX-TZ 3` / `%basis cc-pvdz 3 sto-3g` on ghost-H3 alone.
- Positrons: atoms with no basis get no positron functions. Two atoms of one element with different positron bases are passed to pyscf_neo_cc by
  its per-centre labels.
- Electrons (`%basis`, mirrored): every atom needs a basis, so give one basis without atoms for the rest, e.g.
  `%basis aug-cc-pvtz H cc-pvdz` or `%basis cc-pvdz {1 2} 6-31g`. An atom given its own basis by number is
  labelled `<symbol>@<n>` in the PySCF molecule. With DF, the fitting sets are matched element by element
  (aug-cc-pvtz-jkfit for the H atoms above, cc-pvdz-jkfit for the rest), and the one-positron correlation set
  defaults to aug-cc-pvtz-ri (the library's choice for a mixed basis); the stage warns so you can check.
- Verified 2026-10-14 (HF): `PSX-DZ H` = `PSX-DZ {2}` (LiH, -7.973972632517 Ha);
  `PSX-DZ H PSX-TZ Li` (-7.979453854133); `%basis cc-pvdz O sto-3g H` and
  `%basis cc-pvdz 2 sto-3g` match PySCF with the same per-element / labelled basis to all printed digits.

### 8.3 Removed options

These are errors or ignored; write the replacement instead. The resolved positron atoms are printed in the stage
summary (e.g. `Pos Atoms: N1, N2, O5, O6`); positron functions can be kept off atoms where they are not needed
with atoms on the `%pos_basis` line, which is also the workaround for the library's huge `eri_ep` tensor at
start-up (section 16).

| Removed | Write instead |
|---|---|
| `%pos_atom`, `%pos_basis_map` block, `%pos_basis` as a Python dict or list | atoms and bases on the `%pos_basis` line (section 8.2), e.g. `%pos_basis PSX-DZ H PSX-TZ N` |
| `%psx_set` | nothing: PSX-DZ/TZ/QZ are always the openLOWDIN sets |
| `! WOPT`, `! WRAPPER_OPT`, `! INTRINSIC_OPT`, `! ZOPT`, `! ZMAT_OPT` | `! OPT` with `%opt_engine wopt`, `geometric` or `zopt` (section 10) |
| `! MORSE_SCAN`, `! MORSESCAN`, `! HARMONIC_SCAN`, `! HARM_SCAN`, `! HARMONICSCAN` | `! SCAN` with `%scan_fit morse` or `harmonic` (section 12) |
| `! IRC`, `! NEB`, `! CI_NEB`, `! SADDLE`, `! TS_OPT` | `! TS` with `%ts_mode irc`, `neb` or `ts` (section 9.2) |

## 9. Run types

| Run type | What it does | Result passed to the next stage |
|---|---|---|
| `SP_ENERGY` | mean field, then CCSD / CCSD(T) if requested | input geometry |
| `GRADIENT` | the above plus the analytic nuclear gradient (Hartree/Bohr, printed) | input geometry |
| `OPT` | geometry optimization (section 10) | optimized geometry |
| `SCAN` | single points along one coordinate, no fit (section 12) | lowest-energy scanned geometry |
| `FREQ`, `ANFREQ`, `NUMFREQ` (v04.05) | nuclear Hessian, harmonic frequencies, zero-point energy and normal modes (section 9.1) | input geometry |
| `TS` (v04.05) | transition-state search (geomeTRIC P-RFO) and a frequency check; with `%ts_mode irc` the reaction path, with `%ts_mode neb` a climbing-image NEB (section 9.2) | transition-state geometry (`neb`: highest interior image, a TS guess) |


### 9.1 Harmonic frequencies (`! FREQ`, v04.05)

`FREQ` computes the nuclear Hessian at the stage geometry and diagonalizes it: harmonic frequencies, reduced
masses, force constants, zero-point energy (ZPE) and normal modes. Run it at a minimum found with the same
method, basis and integrals, usually as the stage after an OPT, which passes its geometry on:

```
! OPT CCSD(T) POSITRON DF
%jobname opt
...geometry...
%next
! FREQ CCSD(T) POSITRON DF
%jobname freq
```

`! OPT FREQ` on one line (as in ORCA) does the same without writing the second stage (section 4).

**Analytic or numerical: `FREQ`, `ANFREQ`, `NUMFREQ`** (as ORCA's `Freq`, `AnFreq`, `NumFreq`):

| `!` word | Hessian |
|---|---|
| `FREQ` | the analytic Hessian when the method has one, otherwise a numerical one |
| `ANFREQ` | the analytic Hessian only; an input error for a method without one |
| `NUMFREQ` | always numerical: from analytic gradients where they exist, else from energies (even when an analytic Hessian exists, e.g. to check it) |

Analytic routes are electronic HF/DFT and conventional restricted CCSD with one positron and one
positron-basis carrier. `NUMFREQ` explicitly selects numerical differences on those routes too.
Its default gradient-difference step is 0.005 bohr.

**How the Hessian is obtained** (the stage summary and the log name the route):

| Stage | Hessian | Cost (N atoms) |
|---|---|---|
| ELECTRONIC HF / DFT | PySCF's analytic Hessian | about one gradient |
| POSITRON CCSD CONV, one positron and one carrier | public analytic CCSD Hessian | one SCF/CC/Lambda, 3N response columns, one reference gradient |
| other stages with an admitted analytic gradient | central differences of the analytic gradient, step 0.005 bohr | 6N + 1 gradients |
| admitted energy-only routes, including conventional frozen pair CCSD or pair triples | second differences of energies, step 0.01 bohr (warned) | 1 + 18N² energies |

- `%freq_step <bohr>` changes the step (1e-4 to 0.05).
- Numerical CC equations are converged tighter in FREQ stages (energy 1e-10, amplitudes 1e-8), because differences need
  converged gradients.
- Accuracy, water HF/6-31G against the analytic Hessian: gradient differences within 0.12 cm^-1, energy
  differences within 0.5 cm^-1.
- The analytic positron route uses the same converged CCSD and Lambda for its Hessian and reference
  gradient. It supports the admitted frozen-core cases and explicit positron mass. DF, multiple carriers,
  pairs, triples and quantum-site replacement retain numerical frequencies. See
  [analytic Hessian details and restart costs](ANALYTIC_HESSIAN.md).
- FREQ runs in the ELECTRONIC and POSITRON sectors. PROTON / CNEO stages are refused, because moving a quantum
  nucleus' basis centre is not a vibration. Ghost atoms (no mass) are refused too; constraints and pins are
  ignored.

**What is reported.**
- The log gives the reference energy and its largest gradient component. Above 1e-3 Ha/bohr it warns that the
  geometry is not a stationary point and the frequencies are not meaningful.
- The frequencies in cm^-1, with reduced masses (amu) and force constants (mdyn/A). Translations and rotations
  are projected out (5 for a linear molecule). Imaginary frequencies are printed negative and counted.
- ZPE in Ha, kJ/mol and kcal/mol, and energy + ZPE.
- Masses: the most abundant isotope of each element (H 1.007825, Li 7.016004, Br 78.918338 amu), as Gaussian and
  ORCA use.
- `<job>_freq.txt`: all of the above with the normal modes and the Cartesian Hessian (Ha/bohr^2); a decimal comma
  with `--decimal-comma`.
- `<job>_freq.molden`: the normal modes in Molden's vibration format; open it in Jmol, Avogadro or Molden to
  animate them.

**Cluster runs.** With stage records on (a SLURM job, `--resume` or `--stage-records`), every finished
displacement is kept in `<job>_freq_points.json`. A requeued FREQ stage reuses them and computes only the missing
ones. A local run writes no such file.

**Diatomics.** For a diatomic `FREQ` gives the harmonic frequency at the minimum, the quantity a harmonic-fitted scan
fits (the fit also feels the anharmonicity inside its range), from 13 gradients. A Morse-fitted scan still
gives the anharmonicity omega_e x_e.

### 9.2 Transition states (`! TS`, v04.05)

All three are written `! TS` (or `! OptTS`) and `%ts_mode` chooses the calculation: `ts` (the
default, the P-RFO search), `irc` or `neb`. Below, "IRC" and "NEB" mean `! TS` with `%ts_mode irc` / `neb`.

All three use geomeTRIC (1.1.1, installed with the wrapper's environment) driven by the wrapper's own energies and
gradients, so they work for every stage the wrapper can optimize: ELECTRONIC and POSITRON, DF or CONV, HF, MP2,
CCSD and CCSD(T), one or two positrons, frozen core. (PySCF and pyscf_neo_cc have no transition-state search of
their own.) PROTON / CNEO stages, ghost atoms and systems with fewer than 3 atoms are refused.

**`! TS`: transition-state search (the efficient default).** geomeTRIC's partitioned rational-function
optimization (P-RFO), which walks uphill along one direction of negative curvature and downhill along all
others, from a guess geometry close to the transition state.
- The Hessian is what makes it efficient and reliable. By default (`%ts_hessian auto`) the wrapper computes it
  once at the start, using the same admitted analytic or numerical route as FREQ. Numerical gradient
  differences need 6N + 1 gradients; checkpointed restarts recheck their reference. geomeTRIC then updates
  the Hessian every step, so each step costs one gradient.
- `%ts_hessian` gives geomeTRIC's other choices:

  | Value | Hessian | When |
  |---|---|---|
  | `auto` (default) | the wrapper's, at the start, then updated | nearly always |
  | `first` | geomeTRIC's own finite differences at the start, then updated | same cost as `auto`, for comparison |
  | `each` | geomeTRIC's finite differences at every step | difficult cases (much more expensive) |
  | `stride:N` | geomeTRIC's finite differences every N steps | in between |
  | `never` | geomeTRIC's guess, updated | cheapest, only for very good guesses |
  | `file:<path>` | read from a numpy text file (3N x 3N, Ha/bohr^2), then updated | reuse, e.g. `<job>_initial_hessian.txt` |

- After the search, a frequency check at the result (`%ts_freq true`, the default) must show exactly one imaginary
  frequency: then it is a first-order saddle point, and its normal mode (mode 1 of `<job>_ts_freq.molden`) is the
  reaction coordinate. The log warns when there is none (a minimum) or more than one.
- The step limit is `%opt_max_cycles` (default 100). A TS stage passes the transition-state geometry on.

**IRC (`%ts_mode irc`): intrinsic reaction coordinate.** From a transition state (usually the TS stage just before it), geomeTRIC
follows the steepest-descent path in mass-weighted coordinates downhill in both directions (`%irc_direction`). It
shows which two minima the transition state connects. The starting Hessian is computed as for TS; a Hessian
already computed at the same geometry with the same settings in this job (the TS stage's frequency check) is
reused, so TS then IRC costs one Hessian. The path goes to `<job>_irc.xyz` and its energies to
`<job>_irc_energies.txt`; the log gives both ends relative to the highest point.

**NEB (`%ts_mode neb`): nudged elastic band, to find a guess.** When no good guess exists, geomeTRIC's climbing-image NEB
relaxes a chain of images between two minima: the stage geometry (reactant) and a `%neb_product` block (product;
same atoms, same order, xyz in Angstrom, closed with `end`). The product is aligned onto the reactant and the
starting path is a linear interpolation; `%neb_images` sets the number of images (default 11). One gradient per
moving image per step. The highest interior image is passed on (and written to `<job>_ts_guess.xyz`), so the
usual workflow is NEB, then TS, then IRC:

```
! TS HF ELECTRONIC CONV
%ts_mode neb
%jobname neb
%neb_images 9
* xyz 0 1
...reactant...
*
%neb_product
...product...
end
%next
! TS HF ELECTRONIC CONV
%ts_mode ts
%jobname ts
%next
! TS HF ELECTRONIC CONV
%ts_mode irc
%jobname irc
```

- The end points should be minima: optimize them first, or `%neb_optimize_ends true` (geomeTRIC's end-point
  optimization). The log warns when an end point is above the highest interior image.
- A straight line between the end points must not pass atoms through each other (for example a hydrogen moving
  along a linear molecule, as in HCN -> HNC); then start the TS search from a hand-made guess instead.

**Tests (HF/6-31G and 6-31G\*, against PySCF + geomeTRIC run directly):**
- HCN <-> HNC from a rough guess: 7 gradients to the transition state; energy (-92.7236997 Ha), geometry and
  imaginary frequency (1245.8i cm^-1) equal to PySCF + geomeTRIC's; the IRC ends at the separately optimized HCN
  and HNC minima (barriers 65.647 and 57.270 kcal/mol).
- NH3 inversion: NEB (9 images, 4 cycles, barrier 6.5 kcal/mol), then TS in 2 gradients: the planar transition
  state with one imaginary frequency (976i cm^-1).

## 10. Geometry optimization

| | `AUTO` / `INTRINSIC` (geomeTRIC) | `WOPT` (built-in BFGS) |
|---|---|---|
| Algorithm | geomeTRIC in internal coordinates via `pyscf_neo_cc.geomopt.optimize` | Cartesian BFGS with backtracking line search, rigid-body projection |
| Warm start between steps | yes (faster) | no: fresh SCF/CC at every geometry |
| Sectors | POSITRON, PROTON; ELECTRONIC from v04.05 (PySCF's driver) | all (up to v04.04 ELECTRONIC always used it) |
| `%constraints` | yes; a target value uses geomeTRIC `$set`, otherwise `$freeze` | yes (Lagrange projection + Newton restoration every step) |
| `%pin`, `%pinnings` | yes (`$freeze xyz` / `x`, `y`, `z` after moving to the pinned value) | yes |
| `%equivalences` block | start geometry symmetrized only (warning) | enforced every step |
| `%equivalence` (bond pairs) | not enforced (warning) | exact constraint r(a1,a2) = r(b1,b2) |
| Checkpoint | `<job>_checkpoint.xyz` (trajectory, every step) | `<job>_checkpoint.xyz` (last cycle) |
| Convergence | geomeTRIC defaults | the same five criteria as geomeTRIC: dE < 1e-6, rms grad < 3e-4, max grad < 4.5e-4, rms step < 1.2e-3, max step < 1.8e-3 |

**Gradients: analytic or numerical.** An optimizer needs the gradient (the force on every atom). An
*analytic* gradient is computed from a derived formula, at a cost of one to a few energies. Without one,
the wrapper makes a *numerical* gradient: it moves every coordinate by +-0.0015 A and takes the energy
difference, 6 energies per atom per step (24 for (LiH)2); the result is as good only while every one of
those energies is converged. Which methods have analytic gradients (v04.04):

| Sector | Analytic | Numerical only |
|---|---|---|
| POSITRON | HF, CCSD, CCSD(T), MP2 | Qualified POSHF uses its stationary analytic gradient and geomeTRIC for `OPT`; explicit `WOPT` remains available. The actual HF fitting metrics must be admitted before SCF. |
| ELECTRONIC | HF, DFT, CCSD, CCSD(T) (closed shell), MP2 (CONV) | DF-MP2, open-shell CCSD(T); from v04.05 also DF-CCSD / DF-CCSD(T) (PySCF's formula is 3-5e-5 Ha/Bohr off the DF energy's derivative) |
| PROTON | HF, DFT, MP2, CCSD / CCSD(T) CONV; with CNEO also DF CCSD / CCSD(T) (v04.04) | plain-NEO DF CCSD / CCSD(T) (needs the library's DF-NEO setup; fix proposed) |

The stage summary and `<job>_summary.txt` name the engine that actually runs: `WRAPPER (numerical
gradients: ...)`, `WRAPPER (electronic sector)` up to v04.04, and `geomeTRIC (analytic gradients)` /
`geomeTRIC (numerical gradients)` for electronic stages in v04.05. The log says "numerical gradients" at
every step.

Common to both engines:

- **Energy components (v04.05).** At the final geometry the log prints an `[Energy Components]` block for
  the DF energy and for the `%conv_check` / CONV energy: the mean field (HF), MP2 or CCSD with its
  correlation energy, the (T) correction and CCSD(T). The summary stores the same numbers. So a CCSD(T)
  optimization also gives its CCSD energy at the optimum, with no extra single point. Up to v04.04 only
  the total was kept.
- **Electronic / positronic energy partition (v04.08, POSITRON).** Each of HF, the MP2 or CCSD
  correlation and (T) is split into an electronic part (only electron and nuclear integrals: E_nuc, T_e,
  V_ne, V_ee; the electron-only correlation 2 f(ee)_ia t_ia + sum [2(ia|jb) - (ib|ja)] (t_ijab + t_ia t_jb),
  with the run's own DF or exact integrals; the eee triples) and a positronic part (every term with a
  positron integral: T_p, V_np, V_pp, V_ep; the rest of E_corr; the eep/epp/ppp triples). The table goes to
  the log of a single point and next to the `[Energy Components]` block (DF and CONV) of an optimization,
  and to the summary file. Both parts use the orbitals of the complex, so the electronic part is not the
  energy of the bare molecule or anion. Not for `%pos_spin` (open-shell pair) stages. A line under the
  table lists the (T) terms by type (eee, eep and, with a positron pair, epp, ppp). `%mp2_partition true`
  (POSITRON CC stages) adds the DF-MP2 split of the same POSHF (no new SCF) as an extra row, not summed
  into the Total; positron MP2 exists only with DF, so on a CONV energy that row uses fitted integrals. It
  runs at every point where the partition is computed (each WOPT step included). Tests:
  `tests/test_new_library.py::test_energy_partition_adds_up`, `test_mp2_partition_and_triples_terms`.
  - geomeTRIC: the components come from its fresh final single point.
  - WOPT and ZOPT: they come from the optimizer's own evaluation at that geometry.
  - An energy reused from a ZOPT history file (restart) has no components; the log says so.
- **Final energy.** After the optimization the energy is recomputed by a fresh single point
  at the final geometry. The CC object handed to geomeTRIC keeps its seed-geometry energy
  (library behavior), so it is never used for reporting.
- **Memory.** Before that final single point, the seed-geometry SCF/CC objects are freed.
  Two OOM kills on 2026-09-21 came from both generations being in memory at once.
- **`%conv_check true`** adds an exact-integral single point at the final geometry and
  reports the DF error in the summary. Off by default because it OOM-killed several large
  n_pos=1 jobs.
- **Two positrons with frozen core on geomeTRIC** hit library bug #2 (section 16); the stage
  summary warns. Use `%opt_engine wopt` or `%frozen 0`.

### 10.1 ZOPT: gradient-free Z-matrix optimization (v04.01)

`! OPT` with `%opt_engine zopt` minimizes the energy over the **named variables of the Z-matrix only**, using
energies and no gradients. A variable used in several rows moves all of them at once, so the
symmetry written into the Z-matrix is kept exactly. Numbers written in the Z-matrix stay
fixed. Benzene written with `CC` and `CH` is a two-variable problem, and the result is
exactly D6h:

```
! OPT CCSD POSITRON DF
%opt_engine zopt
* zmat 0 1
C
C,1,CC
C,2,CC,1,120.0
C,3,CC,2,120.0,1,0.0
C,4,CC,3,120.0,2,0.0
C,5,CC,4,120.0,3,0.0
H,1,CH,2,120.0,3,180.0
H,2,CH,3,120.0,4,180.0
H,3,CH,4,120.0,5,180.0
H,4,CH,5,120.0,6,180.0
H,5,CH,6,120.0,1,180.0
H,6,CH,1,120.0,2,180.0

CC = 1.390
CH = 1.084
*
```

Rows may be separated by spaces or commas; `Variables:` / `Constants:` header lines are skipped.

**A Z-matrix with numbers only (v04.07).** If no value is a named variable, ZOPT does not stop: bond lengths
that join the same pair of elements and have the same value (to 1e-4 A) become one variable each (`rCC`,
`rCH`; `rCH_1`, `rCH_2` when one pair has several values), and a warning lists the assignment. Angles and
dihedrals stay fixed (equal angles are not tied: in a ring the ring angles and the H angles can both be 120
without being equivalent). The benzene above written with `1.397` / `1.084` therefore gives the same
two-variable D6h problem. Check that the bonds sharing a variable really are equivalent; to choose yourself,
name the variables. Note that a `Constants:` header does not freeze the names under it: write a fixed value as
a number in the row.
In a ring, consecutive ring atoms need dihedral **0**, and the hydrogens pointing out of the
ring need **180**. With the two values swapped the same rows build an open zigzag chain, with
a C-H pair 0.31 A apart; the close-atom check (section 15) stops such an input.

**How it works.** Every energy computed is remembered, both steps that lowered the energy and
steps that raised it:

1. Start point, then one step up and one step down in each variable, plus one diagonal per
   pair: (n+1)(n+2)/2 energies for n variables (6 for benzene).
2. A quadratic model E(CC, CH, ...) is fitted to the remembered points **near the best one**.
3. The next step goes to the model's minimum, limited to a trust radius. The radius doubles
   after a step that did as well as the model predicted and halves after a poor one (an
   energy rise is also information: it goes into the next model).
4. If there are too few nearby points for a model, the missing points of a smaller pattern
   around the current point are computed.
5. **Converged** when a model built from points within a few tolerances of the best point
   puts the minimum within `%zopt_tol_bond` / `%zopt_tol_angle` of it, and predicts less
   than `%zopt_etol` still to gain. One last energy is then computed at the model minimum.

This is Powell's quadratic-model trust-region idea (UOBYQA). It suits a handful of variables,
because the model needs (n+1)(n+2)/2 points: 3 for 1 variable, 6 for 2, 10 for 3, 28 for 6.
Above about 6 variables, `! OPT` is usually cheaper. Use the symmetry to keep the count low:
one name for every symmetry-equivalent value, and a dummy atom on a symmetry axis where that
removes a dependent dihedral (NH3: 2 variables with a dummy, 3 without). Bonds are kept above 0.3 A and angles
within 1-179 degrees; dihedrals are unrestricted.

**Restart.** Each energy is appended to `<job>_zopt_history.json` as soon as it is computed.
Rerunning the same input (same method, bases, charges, Z-matrix...) reuses every stored
energy, so a job killed by the scheduler continues where it stopped. A history file from a
different calculation is ignored with a message. `%zopt_restart false` ignores it too.

**Not used by ZOPT:** `%constraints`, `%pin`, `%pinnings` and `%equivalence(s)` (warning).
Fix a value by writing it as a number, and make two values equal by giving them the same name.

**Outputs:** `<job>_zopt.txt` (every energy with its variables and why it was computed),
`<job>_final.zmat` (the Z-matrix with optimized values), `<job>_final.xyz`,
`<job>_summary.txt`, `<job>_profile.txt`, and the history file above. The next stage
continues from the optimized geometry.

**Cost compared with the gradient engines:** see `TEST_REPORT.md`, section ZOPT.

#### Hybrid ZOPT: optional analytic gradients (v04.04)

`%zopt_gradient_every N` (off by default) adds analytic gradients where each one replaces several
energies: at the start and at the centre of every rebuilt local model (the "-" samples of the pattern are
dropped), at the best point to confirm convergence (instead of re-sampling a small pattern), and on every
N-th model step. The Cartesian gradient is turned into Z-matrix variables by the chain rule, so the
symmetry written in the Z-matrix stays exact. Each gradient adds one equation per variable to the model.
`%zopt_final_gradient true` computes one analytic gradient at the result and prints it per variable with
the energy change a tolerance-sized move would give ("stationary within the tolerances" or not).

Both need an analytic gradient (section 10; an error otherwise: a numerical one costs 6 energies per atom).
When it pays, measured (a gradient costs 2.6-3.8 energies for positron CCSD; averages over 10 test
functions, cost relative to plain ZOPT):

| Variables | Plain ZOPT | Hybrid (every 3) | Hybrid / plain |
|---|---|---|---|
| 2 | 14 E | 7 E + 3 G | 1.09-1.38 (costs more; warning) |
| 3 | 29 E | 14 E + 4 G | 0.84-1.01 |
| 4 | 43 E | 19 E + 5 G | 0.70-0.83 |
| 5 | 57 E | 27 E + 4 G | 0.66-0.74 |
| 6 | 88 E | 35 E + 5 G | 0.55-0.62 |

On molecules:

| System | Plain ZOPT | Hybrid ZOPT | Gradient engine |
|---|---|---|---|
| NH3 + e+, CCSD, 3 variables | 24 E, about 88 s | 10 E + 3 G, about 81 s | geomeTRIC 4 steps, **60 s** |
| H2O2, electronic CCSD, 4 variables (flat torsion) | 60 E, **not converged** | 34 E + 8 G, converged | WOPT 23 steps (a gradient costs only 1.6 E here), converged |
| CH3O- + e+, CCSD, 4 variables (a gradient costs 3.9 E here) | 35 E, 333 s | 14 E + 3 G, **264 s** | geomeTRIC 6 steps, **257 s** |

(NH3 ZOPT times: measured totals of 101 s and 94 s minus the about 13 s final-gradient check both runs made.)

So: the hybrid is cheaper than plain ZOPT from 3-4 variables and more reliable on flat surfaces; at 4
variables it matched geomeTRIC (CH3O-: 264 s against 257 s); a gradient engine is still faster when it
needs very few steps (NH3: 4). Choose the hybrid when the Z-matrix symmetry
matters and there are 4 or more variables. Memory: any gradient (hybrid, or the final check) needs as
much memory as the gradient engines (NH3 + e+: 0.55 GB against 0.30 GB for energy-only ZOPT).
Tests: `tests/test_zopt_hybrid.py`; runs in `tests/zopt_compare/hybrid_2026-09-24/`.

## 11. Constraints, pinnings and equivalences

All act only in `OPT` stages; elsewhere they are ignored with a warning.

### 11.1 `%constraints` (internal coordinates)

```
%constraints
bond 1 2              # keep the current 1-2 distance   (alias: distance)
bond 1 3 1.10         # drive 1-3 to 1.10 A and keep it
angle 2 1 3 104.5     # angle 2-1-3 (vertex = middle atom)
dihedral 1 2 3 4 180  # IUPAC sign
end
```

The target is the number after the atom list; a trailing integer is a target, not a fourth
index (v03.17 misread it). The dihedral value and its gradient were both wrong in
twoposBrorsen and v03.17 (the WOPT engine could not hold a dihedral); they are now the
IUPAC definition and the exact gradient, checked by finite differences.

### 11.2 `%pin` (whole atoms)

`%pin 1,3` keeps atoms 1 and 3 exactly where they start. In v03.17 the gradient was zeroed
*before* the rigid-body projection, which re-introduced motion, so pinned atoms drifted. The
test suite checks they stay put.

### 11.3 `%pinnings` (single coordinates, twoposBrorsen / v5 syntax)

```
%pinnings
1 0 0.0     # atom 1, x = 0.0     (axis 0/1/2 or x/y/z)
1 1 0.0     # atom 1, y = 0.0
2 z 1.5     # atom 2, z = 1.5
end
```

The coordinate is **set to the value first** (the start geometry moves; each moved atom is
printed), then held fixed. For example, pinning `1 2 0.0` on `F 0 0 -1.85 / Cl 0 0 1.85`
moves F to the origin and halves the starting distance. This is the twopos/v5 behavior.

### 11.4 `%equivalences` (coordinate equivalences, twopos / v5 syntax)

```
%equivalences
2 1 1.0 1.0 -1.0   # atom 2 = atom 1 * (1, 1, -1): mirror through the xy-plane
end
```

The target atom's coordinates are rebuilt from the source atom; its force is folded onto the
source. WOPT enforces this every step; geomeTRIC only symmetrizes the start geometry.

### 11.5 `%equivalence` (bond pairs, v03.17 syntax)

`%equivalence 1,2,3,4` holds bond 1-2 exactly equal to bond 3-4 (WOPT engine only). It is a
real constraint, r(1,2) - r(3,4) = 0, handled like `%constraints`, so an unequal start geometry is
made equal first (the adjustment is printed). v03.17 only averaged the gradient along the two
bonds, which never removed an unequal start; in a test the two O-H bonds of water ended at 1.094
and 0.926 A.

### 11.6 Combining them

Pins, pinnings and equivalence targets are frozen coordinates: the constraint correction and the
constraint projection work only on the remaining coordinates, so a pinned atom never moves even
when a bond/angle/dihedral constraint involves it. All combinations are tested in real
optimizations (section 18).

### 11.7 Old 0-based inputs

twoposBrorsen 00.03-00.16 numbered atoms from 0 in these blocks. A block that contains atom
number `0` cannot be 1-based, so v04.00 reads that whole block as 0-based and warns. Write
1-based numbers in new inputs.

## 12. Potential-energy scans, Morse and harmonic fits

A scan is a series of single points along one coordinate (no gradients, so it also works
where analytic gradients fail or run out of memory). `! SCAN` computes the points; `%scan_fit` chooses what is
done with them (this stage's points only, never pooled).

| `%scan_fit` | Fit | Extra output |
|---|---|---|
| `none` (default) | none: the energies are the result | - |
| `morse` | `E(r) = De (1 - exp(-a (r - re)))^2 + E0` (4+ points) | `<job>_morse_fit_summary.txt` |
| `harmonic` | `E(x) = E0 + k/2 (x - xe)^2` (3+ points) | `<job>_harmonic_fit_summary.txt` |

The points are identical in all three (tested), so a `SCAN` can be fitted later from its
`.scan` table without recomputing.

### 12.1 What to scan

| Scan line | Coordinate | What moves |
|---|---|---|
| `%scan r_min r_max n` (one line) or `fragment r_min r_max n` | gap between two fragments (A) | the atom list is split at its largest sequential gap; both fragments move rigidly apart, symmetrically |
| `bond i j start end n` (or `i j start end n`) | distance i-j (A) | atoms i and j, symmetrically about their midpoint; all others fixed |
| `translate i k axis start end [n=5]` | displacement (A, not a distance) | atoms i..k rigidly along x/y/z |
| `sym_zmat var start end n` | Z-matrix variable (A for a bond length, deg for an angle or dihedral) | the Z-matrix is rebuilt; every row using `var` changes together |

**v04.08: coordinate on the `%scan` line, segments and listed values** (ORCA-like). The coordinate is
`bond i j`, `angle i j k`, `dihedral i j k l` (Cartesian geometry, 1-based atoms) or a Z-matrix variable by
name (`%scan rcc`, `%scan theta`; the old `sym_zmat` spelling still works). The points follow it on the same
line, or in a block of segments:

```
%scan bond 1 2 1.2 1.6 5          # one segment: start end n_points (both ends included)
%scan dihedral 3 1 2 4 [-180 -90 0 60 120]   # listed values
%scan rcc                         # a block of segments, each 'start end n_points'
1.10 1.30 3
1.30 2.00 8                       # 1.30 is not repeated where a segment starts at the previous end
end
```

| Coordinate | Unit | What moves |
|---|---|---|
| `bond i j` | A | atoms i and j, symmetrically (as before) |
| `angle i j k` | deg, (0, 180] | atom k only, turned about j in the i-j-k plane |
| `dihedral i j k l` | deg | atom l only, turned about the j-k axis (IUPAC sign) |
| Z-matrix variable | A or deg | the Z-matrix is rebuilt: every atom placed through that variable moves |

To move a whole group with an angle or dihedral, use a Z-matrix and scan its variable. `angle` / `dihedral`
with a Z-matrix geometry is an error. Each segment needs at least 2 points. Fits work on the joined points
(the 1-D frequency needs a bond or fragment scan; an angle has no reduced mass). Tests:
`tests/test_parser.py::test_scan_v0408_forms`, inputs `tests/inputs/scan_v0408/`.

- One scan per stage; put several scans in separate `%next` stages.
- `%scan_print eda` (v04.08) also writes `<job>_scan_eda.txt`: one line per point with the coordinate and
  every energy component of the method, in Hartree. HF terms `E_nuc T_e V_ne V_ee` (POSITRON: also
  `T_p V_np V_pp V_ep`, then `HF_e HF_p`), `E_HF`; MP2: `Ecorr_MP2 E_MP2`; CCSD: `Ecorr_CCSD E_CCSD`;
  CCSD(T): `E_(T) E_CCSD(T)`. Each method adds only its own component and total (the HF columns are not
  repeated). POSITRON stages split every correlation component into `_e` / `_p` (electronic / positronic,
  as the energy partition of section 10); no sum columns, the parts add up to the component. In a CC stage the MP2 columns are the MP2 the CC
  itself starts from (ELECTRONIC: PySCF's first-guess amplitudes, same integrals) or, for POSITRON, a DF-MP2
  on the same POSHF (pyscf_neo_cc's CC keeps none; one extra DF-MP2 per point); PROTON gives `nan` there.
  Test: `tests/test_features.py::test_scan_print_eda`.
- A one-line `%scan` needs one of the three scan run types on the `!` line. A `%scan` block
  turns an `SP_ENERGY` stage into a plain `SCAN` (as in twopos), but is ignored with a warning
  when the `!` line explicitly asks for `OPT` or `GRADIENT`.
- Every point prints its geometry (non-fragment scans) and energies.

### 12.2 Morse fit (`%scan_fit morse`)

Reports re, De (Ha and kcal/mol), a, E0, the dissociation asymptote E0+De, R^2 and the largest
residual. Needs at least 4 points. Choose a range that spans the minimum and reaches well
into the repulsive and the dissociating parts. For a length coordinate with a defined reduced
mass (below), it also reports the Morse **omega_e** = a sqrt(2 De / mu) and **omega_e x_e** =
omega_e^2 / (4 De), both in cm-1.

Reliability warnings (v04.08): a Morse fit can converge to a wrong re / a with a good-looking R^2 when few
points sample the well (seen on wide Cl2 2e+ scans, `wrapper/bug_reports/2026-10-15_morse_fit_no_reliability_check.md`).
The fit now warns, in the log and in the summary file, when R^2 < 0.995 or when fewer than 3 points lie
inside the well (E < E0 + De/2); it also starts from a width a taken from the local curvature besides
a = 2.0 and keeps the better fit. Take re and k from a narrow `%scan_fit harmonic` scan, and from a Morse
fit only omega_e x_e and the asymptote.

### 12.3 Harmonic fit (`%scan_fit harmonic`)

A least-squares quadratic through the points gives:

| Quantity | Meaning |
|---|---|
| xe | position of the minimum |
| E0 | energy at the minimum |
| k | force constant (curvature): Ha/A^2 and mdyn/A for lengths; Ha/deg^2 and Ha/rad^2 for angles |
| omega_e | harmonic frequency (cm-1) = sqrt(k / mu) / (2 pi c), for a length coordinate with a defined reduced mass |
| anharmonicity check | how much k changes when a cubic term is added (4+ points) |

A harmonic fit describes only the bottom of the well. Use a **narrow, symmetric range
around the minimum**, e.g. +-0.01-0.03 A for a covalent bond (wider for the soft
positron-bound ion pairs of this project), with 5-7 points. If k changes by more than 2% when a
cubic term is added, the stage warns that the points reach the anharmonic region.

Example, H2 HF/STO-3G (tested against PySCF's analytic Hessian):

| Range | omega_e | k change with cubic | |
|---|---|---|---|
| 0.700 - 0.725 A, 7 points | 5478 cm-1 | 0.09 % | reliable (PySCF analytic Hessian: 5479.5) |
| 0.55 - 0.95 A, 7 points | 5218 cm-1 | 4.6 % | warned: range too wide |

A typical workflow: a Morse-fitted scan over a wide range to locate re and De, then a harmonic-fitted scan
over a narrow range around that re for the force constant and frequency.

### 12.4 Reduced mass and the meaning of omega_e

omega_e is a **1-D estimate along the scanned coordinate**, not a normal-mode frequency of a
polyatomic molecule. The reduced mass used (masses of the most common isotopes) is printed:

| Scan | Reduced mass |
|---|---|
| fragment | the two fragments' total masses |
| bond i j | atoms i and j (other atoms held fixed) |
| translate i..k | the moving group against the rest of the molecule |
| sym_zmat, or any angle | none: no frequency is reported, only k |

For a diatomic (and for two rigid fragments) this is the usual harmonic frequency.

### 12.5 Geometry passed to the next stage

- All three scan types pass the lowest-energy scanned point (v04.05, 2026-09-30), so the next stage can scan
  again around it (a finer grid, or another coordinate) or optimize from it. The fitted minimum is printed.

> twoposBrorsen fitted one Morse curve to the points of **all** blocks in the file, including
> every 2-atom single point. Its own test output shows fits of LiH, water and Li3H3 points
> together (R_e = 0.93 A, D_e = 69 Ha). v04.00 never pools stages.

## 13. Output files

| File | Written by | Contents |
|---|---|---|
| `<input>.out` | every run | full log (flushed line by line) |
| `<job>_final.xyz` | OPT | final geometry |
| `<job>_summary.txt` | OPT | wrapper version, sector, method, engine, bases, positrons, frozen orbitals, final DF / CONV energies (v04.05: also their components: HF, MP2 or CCSD, (T), CCSD(T)), final coordinates |
| `<job>_stage<N>_started.json`, `<job>_stage<N>_done.json` | every stage of a SLURM job, or with `--resume` / `--stage-records` (v04.05) | stage records for `--resume` (section 2) |
| `<job>_hf_electrons.molden`, `<job>_hf_positrons.molden` | `%molden true` (v04.05) | canonical HF / POSHF orbitals with energies and occupations |
| `<job>_ccsd_electrons_natorb.molden`, `<job>_ccsd_positrons_natorb.molden` | `%molden true`, CC stages | natural orbitals of the CCSD one-particle density, with occupations |
| `<job>_{hf,ccsd}_{electron,positron}_density.cube` | `%cube true` | densities on a `%cube_points`^3 grid |
| `<job>_{hf,ccsd}_{electron,positron}_density_map.svg` / `.png` | `%density_map` (v04.05) | density on a plane; `both` adds `<job>_{hf,ccsd}_electron_positron_density_map` with the two side by side |
| `<job>_{hf,ccsd}_esp_map.svg` / `.png` | `%density_map esp` (v04.05) | electrostatic potential of the nuclei and electrons on a plane |
| `<job>_combined_density_map.svg` / `.png` | `%density_map_combine` (v04.07) | maps of several stages side by side (e.g. 0, 1, 2 positrons) |
| `<job>_profile.txt` | OPT | every pairwise distance and angle of the final geometry |
| `<job>_checkpoint.xyz` | OPT | latest geometry (WOPT) or trajectory (geomeTRIC); restart point after a crash |
| `<job>.scan` | SCAN | tab-separated: `Coordinate, HF_Ha, CCSD_Ha, CCSD(T)_Ha, Total_Ha, Integral_Type` (`nan` = not computed) |
| `<job>_scan_points.txt` | SCAN | coordinate and total energy (v03.17 format) |
| `<job>.scan` / `<job>_scan_points.txt` | all three scan types | see above |
| `<job>_morse_fit_summary.txt` | `%scan_fit morse` | re, De (Ha and kcal/mol), a, E0, E0+De, R^2, omega_e, omega_e x_e, or "FAILED" |
| `<job>_freq.txt`, `<job>_freq.molden` | FREQ (v04.05) | frequencies, reduced masses, force constants, ZPE, normal modes, Hessian; normal modes for Jmol / Avogadro / Molden |
| `<job>_ts_path.xyz`, `<job>_geometric.log`, `<job>_initial_hessian.txt` | TS (v04.05) | the search trajectory, geomeTRIC's log, the starting Hessian (numpy text, Ha/bohr^2) |
| `<job>_ts_freq.txt`, `<job>_ts_freq.molden` | TS | frequency check at the transition state (mode 1 = the reaction coordinate) |
| `<job>_irc.xyz`, `<job>_irc_energies.txt` | IRC (v04.05) | the reaction path and its energies (Ha and kcal/mol relative to the highest point) |
| `<job>_neb_path.xyz`, `<job>_neb_energies.txt`, `<job>_ts_guess.xyz` | NEB (v04.05) | the final band, its energies, the highest interior image |
| `<job>_freq_points.json` | FREQ in a SLURM job (or `--resume` / `--stage-records`) | finished displacements, reused after a requeue |
| `<job>_harmonic_fit_summary.txt` | `%scan_fit harmonic` | xe, E0, k, R^2, reduced mass, omega_e, anharmonicity check, or "FAILED" |
| `<job>_zopt.txt`, `<job>_final.zmat`, `<job>_zopt_history.json` | ZOPT (v04.01) | energy history table; optimized Z-matrix; restart file (section 10.1) |
| `<job>_final.xyz`, `_summary.txt`, `_profile.txt` | ZOPT (v04.01) | as for OPT |

`<job>` is `%jobname`; give each stage its own name or its files overwrite each other.

**Wavefunction files for Multiwfn, Jmol, VMD (`%molden`, `%cube`; v04.05).** The files are written at the
stage's final geometry: the optimized one for OPT and ZOPT, the input one for SP_ENERGY and GRADIENT. They
are available in the ELECTRONIC and POSITRON sectors.

- **Levels.** HF files come from every stage. CC files come from CCSD and CCSD(T) stages and use the
  unrelaxed CCSD one-particle density (the (T) correction is not included).
  - Electronic: PySCF's density.
  - One positron: the library's one-particle densities.
  - Two positrons: `positron.density.ccsd_dm1_ao`.
  - MP2 stages and triplet positron pairs write the HF level only.
  - On the old library (PySCF 2.13 fork with the August pyscf_neo_cc), positron systems write the HF level
    only: its CCSD densities of positron systems are not reproducible between identical runs. Use
    pyscf_neo_cc 2026-09-23 (`MCCC_0923_stock`).
- **Cost.** The export runs one more single point, plus the Lambda equations for CC.
- **Electrons and positrons are in separate files**, because the formats know only electrons. Positron
  orbitals and densities are correct. But a program that combines the density with the nuclear charges
  (electrostatic potential, QTAIM, ELF) treats the positron file as electrons, so those analyses are not
  meaningful for it.
- **Natural orbitals** are those of the symmetric part of the CC density (the unrelaxed CC density is not
  symmetric). Their occupations sum to the particle count.
- **Cube boxes.** The electron box reaches 5 bohr beyond the atoms and the positron box 15 bohr, measured
  so the cubes hold the particles: electrons within about 1% at 80 points, positrons 100%. PySCF's default
  3-bohr box held only 19-52% of a positron. The log prints how much of each species every cube holds.

**CHELPG charges (`%chelpg`; v04.05).** Atomic charges fitted to the electrostatic potential at the final
geometry, for the HF and (CC stages) CCSD densities, printed in the `.out` as a table with the fit RMSE. The
wrapper computes them itself (PySCF integrals, no extra program), with Multiwfn's settings (Breneman and
Wiberg):
- fitting points on a 0.3 A grid, in a box 2.8 A beyond the atoms, outside every atom's radius and within
  2.8 A of an atom;
- radii (A): H, He 1.45; Li-C 1.50; N-Ne 1.70; Na-Ar 2.00; heavier atoms the UFF radius / 1.2, as Multiwfn does
  (Br 1.745);
- the potential fitted is that of the nuclei and the electrons, and the charges sum to its charge.

For LiH the fitting points are Multiwfn's (3927). Charges agree with Multiwfn's to about 5e-5; the small
difference is Multiwfn's fast potential code (about 1e-5 a.u. off), the wrapper uses exact integrals.

**Positron systems.** The positron density is left out of the fit: charges that include it are not defined.
The charges therefore describe the electrons as the positron has polarized them, and sum to the charge plus the
number of positrons (0 for e+ LiH). This is what Multiwfn gives for the electron Molden file. Use the positron
density map to show where the positron is.

Example, e+ LiH CCSD/6-31G, PSX-DZ:

```
[CHELPG] CCSD charges at the final geometry: 3927 fitting points (0.3 A grid, 2.8 A box; radii Li 1.50, H 1.45 A)
    #  Atom             q
    1  Li        0.794308
    2  H        -0.794308
  Sum            0.000000
  RMSE (a.u.)    0.007623
```

**Density maps (`%density_map`; v04.05).** Colour-filled maps of the density on a plane, in the style of
Multiwfn's plane maps (rainbow scale, white above and black below the scale, contour lines, atom labels,
axes in bohr), for the HF and (CC stages) CCSD levels. The wrapper draws them itself, with no plotting
library:
- `%density_map electron`: the electron density, scale 0-0.65 a.u. (Multiwfn's default), box 4.5 bohr beyond
  the atoms;
- `%density_map positron` (POSITRON only): the positron density, scale from 0 to its maximum on the plane (in
  1e-3 a.u. when it is small), box 10 bohr beyond the atoms;
- `%density_map both`: the two maps, plus one figure with them side by side;
- `%density_map true`: electron for ELECTRONIC stages, both for POSITRON stages;
- `%density_map esp`: the electrostatic potential (ESP) map, see below.

**Colour-scale limits and colours, on the same line.** After the species come up to two ranges and then a
colour scheme, all optional:

```
%density_map electron 0,0.5 viridis           # electrons, scale 0-0.5 a.u., viridis
%density_map positron 0,3e-3                  # positron, scale 0-0.003 a.u. (shown as 0-3 x 1e-3 a.u.)
%density_map both 0,0.4 0,0.003 viridis       # electrons 0-0.4, positron 0-0.003
%density_map both auto 0,2e-3 18              # electrons at the default 0-0.65, positron 0-0.002; scheme 18
%density_map both 0,0.4 0,0.003 viridis ystep=2   # as above, vertical-axis labels every 2 bohr
```

- A range is `lower,upper` in a.u. (`auto` keeps the default). With `both` give two ranges, electrons first;
  with one species, one range. A scale whose upper limit is below 0.01 a.u. is shown in 1e-3 a.u.
- The colour scheme goes last. It is one of Multiwfn's 20 colour transitions (plane-map option 19), by number or
  name; the colours were sampled from Multiwfn's own colour bars:

  | # | name | # | name |
  |---|---|---|---|
  | 0 | `rainbow_bw` (default) | 10 | `blue_green_red` |
  | 1 | `rainbow` | 11 | `red_green_blue` |
  | 2 | `reversed_rainbow` | 12 | `white_darkred` |
  | 3 | `white_rainbow` | 13 | `black_orange_yellow` |
  | 4 | `spectrum` | 14 | `white_darkgreen` |
  | 5 | `reversed_spectrum` | 15 | `black_green` |
  | 6 | `grey` | 16 | `white_darkblue` |
  | 7 | `reversed_grey` | 17 | `black_blue_cyan` |
  | 8 | `blue_white_red` | 18 | `viridis` |
  | 9 | `red_white_blue` | 19 | `yellow_orange_black` |

  Names ignore case, and `-` equals `_` (`Blue-White-Red`). As in Multiwfn, only scheme 0 marks values outside
  the limits (white above, black below); the others draw them in their end colours.
- `ystep=<bohr>` (anywhere after the species) sets the spacing of the labels on the vertical axis, for every map
  of the stage; by default it is chosen automatically. A spacing that would give more than 60 labels falls back to
  automatic, and the log says so.
- Contour lines follow the scale: the electron defaults (0.01-0.4 a.u.) within the limits, otherwise at the
  colour-bar ticks.
- The log gives each map's maximum on the plane, its scale and the colours.

**Combined figure of several stages (`%density_map_combine`; v04.07).** One figure with maps of different
stages side by side, e.g. the same molecule with 0, 1 and 2 positrons, written by one line in the last stage:

```
%density_map_combine X_0pos X_1pos X_2pos          # electrons of X_0pos, positrons of X_1pos and X_2pos
%density_map_combine X_1pos:electron X_2pos:electron level=hf scale=own
```

- Each entry is a `%jobname` of this stage or an earlier one, optionally `:electron` / `:positron` (default:
  positron for a POSITRON stage, electron for an ELECTRONIC one). Panels go left to right in the order given.
- The named stages keep their final-geometry densities (the export runs for them even without `%density_map`,
  `%molden` or `%cube`; CC stages solve the Lambda equations for the CCSD density).
- `level=ccsd` (default; a stage without a CC density falls back to HF, with a notice) or `level=hf`.
- `scale=shared` (default): one colour scale per species over the figure (the largest of the panels' default
  upper limits); `scale=own`: each panel its own default scale.
- Plane: the first entry's plane (`%density_map_plane` of the combining stage, AUTO by default) is used for every
  panel; a warning is printed when the other stages' atoms differ from the first's by more than 0.1 bohr.
- Output: `<job>_combined_density_map.svg` / `.png` (`<job>` = the combining stage). If a named stage left no
  density (its export failed, or `--resume` skipped it), the figure is skipped with a warning.
- From saved files: the file viewer's option 12 builds the same figure from several Molden / cube files.

**Electrostatic potential maps (`%density_map esp`).** The potential of the nuclei and the electrons,
V(r) = sum_A Z_A / |r - R_A| - integral rho_e(r') / |r - r'| dr', on the same plane as the density maps, for the
HF and (CC stages) CCSD electron densities. For positron systems the positron density is not included, as for
`%chelpg` (an ESP map is made from the electron density, as from the electron Molden or cube file); the subtitle
and the log say so.
- The same options as the density maps: one colour range (default -0.1,0.1 a.u.), a colour scheme (default
  `blue_white_red`, Multiwfn's scheme 8, white at zero) and `ystep=`. For example
  `%density_map esp -0.05,0.05 red_white_blue ystep=2`.
- Contour lines at the colour-bar ticks: positive solid, negative dashed. The thick line is the electron density
  rho = 0.001 a.u., the usual van der Waals surface (Multiwfn's plane-map option 15).
- 200 x 200 points (Multiwfn's default), box 4.5 bohr beyond the atoms. The potential uses exact integrals.
- The log gives the range of the potential on and outside the rho = 0.001 a.u. line (inside it the nuclei
  dominate).

**Several maps in one stage.** Every `%density_map` line adds its own map with its own options, so one stage can
draw the density maps and the ESP map:

```
%density_map both 0,0.4 0,0.003 viridis
%density_map esp -0.05,0.05 ystep=2
```

- Asking for the same map twice in a stage is an input error (the files would overwrite each other), for example
  `both` and `electron`, or two `esp` lines.
- `%density_map false` removes the maps asked for before it in the stage.
- `%density_map_plane` applies to all the maps of the stage.

The plane is `%density_map_plane` (XY, XZ, YZ), or by default the Cartesian plane with the largest spread of
the atoms (XZ for a molecule along z), through the atoms' mean position. Atoms within 0.5 bohr of the plane are
labelled. The maps are SVG files; a PNG copy is made when `rsvg-convert` or ImageMagick is installed (the log
says when it is not: SVG opens in any web browser or Inkscape).

A failed analysis prints a notice and does not stop the job or the other analyses.

**Fitting sets that lack an element (v04.05).** PySCF's cc-pVXZ / aug-cc-pVXZ RI and JK-fit families have no
entries for Li, Be, Na, Mg and K. For the mean field the wrapper uses `def2-universal-jkfit` for those atoms. For
the correlation set of one-positron stages, pyscf_neo_cc completes the named set with an even-tempered set built
from the orbital basis. The stage summary names the sets really used, for example
`correlation = aug-cc-pvtz-ri (library default, <basis>-ri); Li: even-tempered set from the orbital basis`, and
validation warns once. Up to v04.04 the summary said only "library default" and PySCF's "Basis may be available
in basis-set-exchange" warning was printed at every energy evaluation; it meant this substitution, not an error.
A named set that has the element (`%auxbasis_e def2-qzvpp-ri`, `cc-pvtz-ri`) avoids the substitution.

## 14. Resources, memory and clusters

- `%nprocs` sets PySCF's thread count for the stage. It overrides `OMP_NUM_THREADS`, so on a
  scheduler set it to the cores you requested (SLURM `--cpus-per-task`).
- `%memory` is PySCF's working-memory target, not a hard limit. Give the scheduler about 10-20% more.
- `%df_mem_cap` controls pyscf_neo_cc's blocked DF intermediates (default min(32 GB, half the
  RAM)). On a large node, raising it may speed up two-positron gradients.
- Measured memory (peak resident, MaxRSS):
  - The 2026-09-23 library uses 20-37% less memory than the old setup on the same CCSD(T)/aug-cc-pVTZ single
    points, and is usually 10-30% faster.
  - Br2(2-) + 2e+ CCSD(T) energy plus analytic gradient (aug-cc-pVTZ / PSX-TZ, frozen core, 6 threads):
    - new library: 10.5 GB peak (set by the energy; the gradient part stays below 4.6 GB), 1 h 57 min;
    - old library: 142.5 GiB for the same job.
    - Source: `pyscf_neo_cc_2026-09-23/old_bug_recheck_2026-09-26/`.
- Don't use `ulimit -v` (a virtual-memory cap) to protect a machine. PySCF's gradient J/K routine reserves
  about 3.6 GB of address space per thread without using it, so at 10 threads a 34 GB cap stops a job that
  needs 10 GB. Limit resident memory instead (the scheduler's memory request, or a cgroup).
- A multi-stage job runs stages one after another in one process; peak memory is the largest stage's.

## 15. Validation: errors and warnings

Before anything is computed, every stage is checked and summarized. **Errors** stop the run
and are all listed together; **warnings** are printed and the run continues.

Errors include: a non-numeric value where a number is expected; a first stage without
geometry; an unknown geometry block; an undefined Z-matrix variable; an atom number out of
range in any block; an atom on the `%pos_basis` line matching nothing; a malformed or second scan in
one stage; `SCAN` without a scan definition; `sym_zmat` without a Z-matrix or with an unused
variable; triplet pairs when the library lacks `POSSOCCSD`. v04.01 adds: two atoms closer
than 0.5 A (usually a wrong Z-matrix dihedral); ZOPT without a `* zmat` block, with no
named variables, with a variable that has no value or is used both as a bond and as an
angle, or with `%zopt_max_evals` too small for the number of variables.

Warnings include: unknown directives or `!` tokens; missing run type; conflicting
charge/multiplicity; unclosed blocks; 0-based blocks; constraints outside OPT; equivalences
on the geomeTRIC engine; `%pos_mult` with one positron; `%df_aux` where it has no effect;
an RI set given as `%jkfit_aux_e` (v04.02: moved to the correlation role);
one-line `%scan` in a non-scan stage; a `%scan` block ignored by an explicit OPT/GRADIENT;
the two-positron frozen-core geomeTRIC bug.

v04.04 adds:
- **Errors:**
  - decimal commas in `* xyz` coordinates;
  - an unknown `%nuc_basis` or `%epc`;
  - `%cneo_variant` other than ccd / ucd;
  - `%cneo` without a quantum nucleus, or on a library without working CNEO (the older library on the PySCF
    2.13 fork aborts in libxc);
  - MP2 outside ELECTRONIC without DF, on the older library, or with an odd number of positrons;
  - invalid `%pos_mass`, `%quantum_site`, `%nuc_mass` or `%quantum_nuc`;
  - `%zopt_gradient_every` without an analytic gradient.
- **Warnings:**
  - an RI two-positron set for Na and heavier atoms (replaced);
  - PySCF's DZ JK-fit sets for Ga-Kr (no g functions);
  - positron-HF OPT runs WOPT with numerical gradients;
  - electronic DF-MP2 gradients are numerical;
  - `%cneo` / `%cneo_variant` / `%nuc_mass` / `%pos_mass` where they have no effect;
  - CNEO ucd optimizations;
  - particle-mass and basis mismatches;
  - `%conv_check` skipped for positron / NEO MP2;
  - hybrid ZOPT with fewer than 3 variables.
- v04.05 makes an **unrecognized directive an error**. Up to v04.04 it was ignored with a warning, so a
  misspelled setting was silently skipped: `%conv_chek true` meant no exact-integral check. Among the
  project's 871 inputs only one has one, `%%memory` in an old Br2 cluster input. Unknown words on the `!`
  line are still warnings.
- v04.05 adds two warnings:
  - an electronic OPT that runs geomeTRIC with numerical gradients (DF-CC, DF-MP2, open-shell CCSD(T));
  - a broken DZ JK-fit set replaced for Ga-Kr.
- **Pre-flight (after the SCF, before the CC; exit code 3):** a two-positron DF stage that needs the
  gradient stops if a fitting metric has no Cholesky factor (section 7.1).

## 16. Known library limitations and workarounds

These are pyscf_neo_cc behaviors, not wrapper bugs. Reports and reproducers:
- `results/Tests/pyscf_neo_cc_bug_reports/` (old library, August-September 2026);
- `pyscf_neo_cc_2026-09-23/bug_reports_2026-09-25/` (the 2026-09-23 release);
- `pyscf_neo_cc_2026-09-23/old_bug_recheck_2026-09-26/` (the old reports rechecked on the 2026-09-23 release).

**Old library only** (all fixed or avoided in the 2026-09-23 release):

| Problem | When | Status / workaround |
|---|---|---|
| Frozen-core warm start crash: `Size of label 'i' ... does not match` on the 2nd geometry | two positrons, `%frozen` > 0, geomeTRIC | fixed in 2026-09-23 (Cl2(2-) + 2e+ frozen 10 converges). On the old library: `%opt_engine wopt` or `%frozen 0` |
| Unbatched o^3 v^3 triples in the two-positron CCSD(T) gradient | two-positron CCSD(T) gradients on large bases | fixed in 2026-09-23 (blocked triples: Br2 2e+ energy and gradient in 10.5 GB; the old library needed 142.5 GiB) |
| Exact electron-positron tensor at SCF start-up (`eri_ep`) | large electron *and* positron bases, e.g. (OCN)2(2-) | 2026-09-23 skips it when the DF recipe is passed to POSHF, which v04.02+ always does ((OCN)2(2-) + e+: 91 s, 2 GB; without the recipe the library still tries 45 GiB) |
| Stale energy on the CC object after `optimize()` | every geomeTRIC OPT | fixed in 2026-09-23; v04.00+ recomputes the final energy anyway |

**2026-09-23 release:**

| Problem | When | Workaround |
|---|---|---|
| The pair CC's Fock matrix is fitted with `df_aux['auxbasis_e']`; an RI set misfits the core | two positrons with Cl, Br or heavier | v04.04: per-element pair set (JK-fit for Na and heavier). Br2 2e+ with RI: 4.88 Ha too low; Cl2: 4.94 mHa |
| A JK-fit pair set can make the unified e-p fit near-singular; the DF gradient refuses after the full CC | two positrons with C/N, e.g. (CN)2(2-) with aug-cc-pvtz-jkfit | v04.04: RI pair set for H-Ne, and the pre-flight check stops a bad stage 18 s after the SCF |
| One-positron POSHF does not converge near (not at) a symmetric geometry | e.g. e+(LiH)2 0.002 A from D2h | `%opt_engine zopt` (keeps the symmetry); the wrapper warns and continues for HF/CC, MP2 stops |
| Even-tempered nuclear basis names not scaled with the mass for `H*`, `H#` | `%nuc_basis` like `6s6p6d` with isotopes | v04.04 scales them itself (section 7.4) |
| NEO-HF / NEO-KS have no `mf.Gradients()` | PROTON mean-field gradients | v04.04 calls `pyscf_neo_cc._neo.Gradients(mf)` |
| `DFCCSD` on a non-DF NEO-HF: the gradient fails with `KeyError: 'nuclei'` | plain NEO DF-CC gradients | v04.04 uses numerical gradients there (decision pending) |
| Positron density matrix and EOM DF gradient depend on the thread count | many threads | open question to the author |
| Original library lacked a positron HF analytic gradient | positron-HF OPT | The qualified followup now uses analytic POSHF geomeTRIC; missing-library support retains the explicit numerical WOPT fallback (section 10). |

## 17. Migrating

### 17.1 From newBRORSEN v03.17 (localfix / merged)

Every v03.17 input in `results/` parses to the same settings (section 18). Things that
change what a v03.17 input does:

| Change | Effect | To keep v03.17 behavior |
|---|---|---|
| `PSX-DZ`/`PSX-TZ` are the openLOWDIN sets | different positron basis, different energies (section 8.1) | give the old exponents as raw shells |
| `%threads` honoured (v03.17 ignored it) | 34 stages of old v5-style inputs now use their `%threads` value instead of 4 | change `%threads` |
| `%pinnings`/`%equivalences` blocks are active (v03.17 ignored them silently) | old v5-style inputs now pin/symmetrize | delete the blocks |
| v04.05: no sector word = ELECTRONIC (was POSITRON) | a stage with positron/proton directives but no sector word is an error (none of this project's 945 inputs change silently) | write `POSITRON` on the `!` line |
| v04.05: default method HF (was CCSD(T)), default `%basis` cc-pvdz (was aug-cc-pvtz) | none of the project's inputs relies on either default | write the method and `%basis` |
| `%pin` atoms really stay fixed, also next to constraints | WOPT results with `%pin` differ (v03.17 let them drift) | - |
| `%equivalence a,b,c,d` is an exact constraint | the bonds are really equal (v03.17 kept any initial difference) | - |
| Dihedral constraints: IUPAC sign, correct gradient | none in this project's inputs | - |
| ELECTRONIC CCSD(T) uses the (T) gradient | ELECTRONIC CCSD(T) OPT geometries change (v03.17's were inconsistent) | - |
| Morse fragment scan whose fit fails | passes the lowest-energy point (v03.17 passed nothing) | - |
| geomeTRIC constraints with a target use `$set` | the target is reached (v03.17 froze the starting value) | omit the value |

### 17.2 From twoposBrorsen 00.81 / 00.86

| twoposBrorsen | v04.00 |
|---|---|
| `! ...` without run type = `OPT` | `SP_ENERGY` + warning |
| only the literal `DF` = density fitting (`RI` = conventional) | `DF` and `RI` both = density fitting |
| `%frozen auto` = 0 for every positron run; Br = 0 | real frozen core in every sector, Br included |
| `%memory`, `%threads`, `%compute_t2e1p` ignored; `%nprocs` silently ignored | all applied |
| Z-matrix third atom at 180 - angle | correct |
| undefined Z-matrix variable = 0.0 | error |
| no geometry chaining (a block without coordinates ran on H2) | chaining works |
| Morse fit pools every block | per stage |
| `%constraints`: bond crashed, indices used as 0-based | work, 1-based |
| no post-OPT exact check | `%conv_check true` |
| positron OPT crashed (no `solve_lambda`) | works (geomeTRIC or WOPT) |
| one-positron DF correlation aux = `%jkfit_aux_e` | library default (as v03.17); `%auxbasis_e <jkfit basis>` gives twopos's numbers |
| catch-all "Cholesky" fallback reran conventional CCSD, still labelled DF | no silent fallback: the real error stops the stage |
| non-converged SCF marked converged | ARH/Newton retry, then a visible warning |
| without `%df_aux`, two-positron "DF" was exact rank-4 | always real DF (`{}` = library defaults) |
| `.scan` columns `R_Angstrom HF CCSD CCSD(T) Integral_Type` | `Coordinate HF CCSD CCSD(T) Total Integral_Type`, one file per scan stage |
| output `trajectory_<job>.txt`, `optimized_<job>.xyz` | `<job>_checkpoint.xyz`, `<job>_final.xyz`, `<job>_summary.txt` |
| no proton sector | PROTON sector |

Inputs that differ only in these ways run unchanged. The positional `!` line, `%threads`,
`%pos_mult`, `%df_aux`, `* zmat`, all `%scan` forms, `%pinnings`,
`%equivalences`, `%optimizer` and `%key=value` are all understood.

## 18. Tests

Run from `wrapper/unified/`:

```bash
python -m pytest            # basic: no calculations, ~15 s
python -m pytest --fast     # basic + a few small real calculations per feature and sector, ~5 min
python -m pytest --full     # everything (552 tests with v04.05): ~45 min per environment
NEWBRORSEN=newBRORSEN.v04.04.py python -m pytest --fast   # test another version (default: v04.06)
```

From v04.05 the tests read their calculations from input files in `tests/inputs/<area>/` (one `.inp` per
case; its header says what it is and which test uses it). They are ordinary wrapper inputs and can be used as
examples. The tests for v04.04 and earlier (inputs written inside the Python files, old defaults) are frozen
in `tests_v0404/`.

| File | What it checks |
|---|---|
| `tests/test_parser.py` | every directive, alias and value form of both syntaxes; comments; `=` forms; geometry blocks; inheritance; positron basis/atom forms; PSX sets; constraints/pinnings/equivalences incl. 0-based; all scan forms; validation errors and warnings |
| `tests/test_geometry.py` | Z-matrix bond lengths, angles, dihedral values and signs; benzene ring closure; constraint gradients vs finite differences; constraint restoration; projections; pinnings/equivalences; scan geometries; geomeTRIC constraint file |
| `tests/test_corpus.py` | every input in `results/` and in the twopos folders parses (inputs that need a newer library are listed, not failed); for newBRORSEN inputs every shared setting equals v03.17_merged's reading, except the listed intentional differences |
| `tests/test_opt_constraints.py` | real optimizations in both engines: dihedral and bond targets, bond-pair equivalence, pin + angle constraint on a positron system, coordinate pinnings, geomeTRIC bond freeze + pin, geomeTRIC symmetrized start |
| `tests/test_zopt.py` | v04.01 ZOPT: engine on analytic surfaces vs exact minima, restart, bounds, variable types; comma Z-matrix benzene ring; close-atom error; ZOPT input errors; benzene HF/STO-3G vs geomeTRIC; H2O vs OPT, restart and chaining |
| `tests/test_fitting_roles.py`, `tests/test_matched_fitting.py`, `tests/test_new_library.py` | v04.02/v04.03: fitting-set roles and defaults; the 2026-09-23 library's DF recipe on POSHF |
| `tests/test_conv_check_safety.py` | v04.04: results survive a job killed during the `%conv_check` single point |
| `tests/test_masses.py` | v04.04: particle masses on three routes; scaled nuclear sets = the library's isotope sets |
| `tests/test_ghost_atoms.py` | v04.04: ghost atoms in every spelling, geometry form, sector and engine |
| `tests/test_mp2.py` | v04.04: every MP2 route = the library called directly; gradients vs finite differences |
| `tests/test_engine_labels.py` | v04.04: the engine each optimization really uses is the one reported |
| `tests/test_zopt_hybrid.py` | v04.04: hybrid ZOPT (gradient-enhanced model) on test surfaces and molecules |
| `tests/test_pair_aux.py` | v04.04: per-element two-positron set; pre-flight check (new library only) |
| `tests/test_cneo.py` | v04.04: CNEO/CDFT energies = the library; gradients vs finite differences; optimizations |
| `tests/test_v0405.py` | v04.05: electronic geomeTRIC = WOPT minimum in fewer steps, constraints and pins; electronic CC gradients vs finite differences; GRADIENT jobs never optimize; Ga-Kr DZ JK-fit replacement (Br- within 0.01 mHa); `--resume` (skip, rerun on edit, checkpoint restart, SLURM requeue) |
| `tests/test_features.py` | SCF fallback; `%df_mem_cap`; triples options; CLI dry run and failure; Z-matrix vs Cartesian; all scan modes vs independent single points; per-stage Morse fits; chaining after scans; `%df_aux` vs a direct library calculation; triplet pairs vs direct `POSSOCCSD`/`POSSODFCCSD`; `%block_size_a`; dict basis vs map; pins, pinnings and equivalences in both engines; geomeTRIC `$set` target; electronic CCSD(T) gradient vs finite differences; `%conv_check`; a twopos-syntax input end to end |

`tests/README.md` explains, block by block, what each test checks and why it matters.
`tests_retired/` keeps tests removed on 2026-09-23 because they only mattered with the old
pyscf_neo_cc (the numerical regression against v03.17 and the library-bug-#2 warning); its README
gives the reasons.

Results of the latest run: see `TEST_REPORT.md` (v04.05 sign-off run: 552 passed on `MCCC_0923_stock`; 516 passed, 36 skipped on `MCCC`).

## 19. Changelog

**v04.05 (2026-09-26 to 2026-09-30; production release, signed off 2026-09-30):**
- 2026-09-30 (director's requests): `SP_ENERGY` and `ELECTRONIC` are the defaults; the optimizer is chosen
  with `! OPT` + `%opt_engine` (section 4); positron/proton directives without their sector word and unknown `!` words are errors; the
  tests read their inputs from `tests/inputs/` (section 18).
- 2026-09-30: coordinate units, `! Bohrs` / `! Angs` or `%coords bohr|angstrom` (section 6.4).
- 2026-09-30: a stage without a geometry block continues all the settings of the stage before (section 3);
  every scan passes its lowest-energy point on; the default method is HF; the default `%basis` is cc-pvdz;
  the fit is chosen with `! SCAN` + `%scan_fit morse|harmonic`.
- 2026-09-30: `! TS` (or `! OptTS`) with `%ts_mode ts|irc|neb`
  (default ts). `%ts_mode` carries over to following stages, except that after a NEB it goes back to ts, and a
  stage that writes `! TS` / `! OptTS` without a `%ts_mode` line runs the TS search.
- 2026-09-30: two-positron CONV CCSD (`POSCCSD`, e.g. the `%conv_check` single point) that stops just short of
  the library's 1e-9 residual limit after 300 cycles (seen on ClOCN / CNOCN + 2 e+ at |r| = 3-4e-9: the energy was
  lost and the log said "CONV cross-check failed (tcc-factored not converged)") is solved once more to 1e-8 with up
  to 600 cycles; the log says so. The energy error of such a residual is far below 1e-10 Ha.
- 2026-09-30: `make_input.py`, an input maker that asks questions and writes the `.inp` (tests: `tests/test_make_input.py`).
**v04.07 (2026-10-08; development build, not released):** v04.06 plus the `--ui` terminal input builder
(section 2.1), `parse_text(text)` and `--parse-text-json` (the parser and validator on a text, diagnostics as
JSON, no files written). `make_input.py` is retired (`archive/make_input.py`). Tests: `tests/test_ui.py`, inputs
`tests/inputs/ui/`. Spec: `spec-newbrorsen-tui-v0407.md`.
- 2026-10-08: `%next` is accepted (gray template of the stage it closes stops being offered) with **Enter**
  as well as Tab; only Tab worked before, so writing `%next` and pressing Enter left the stage's template
  showing indefinitely.
- 2026-10-08: the start screen gets a **View molden/cube** entry, before Settings, first folding v04.06's
  standalone Molden/cube file viewer into `--ui` as a subprocess hand-off, then (same day, on request)
  reworked into a native `--ui` screen: left pane lists loaded files and settings, right pane previews the
  plot inline (Kitty graphics protocol, iTerm2, or Sixel via ImageMagick/`img2sixel`) or opens it in the
  system image viewer. Numpy/PySCF, which it needs and the fast `--ui` path never imports, load lazily the
  moment it is opened (`ui_load_heavy`/`runpy.run_path`), not on every `--ui` start. Also new: **Compare
  included files**, one figure with a panel per loaded file (any role, own label) side by side, for N-way
  comparisons the fixed electron/positron pairing does not cover (for example the electron density of the
  same system with 0, 1 and 2 positrons attached). `--ui job.molden`/`--ui job.cube` go straight into this
  screen, file preloaded.
- 2026-10-08: fixed three bugs found testing the above on the user's own terminals: a crash on every render
  once a plot was drawn (`WindowRenderInfo.y_offset`/`.x_offset` do not exist; the window's screen position
  now comes from `Renderer.last_rendered_screen.visible_windows_to_write_positions`); wrongly claiming Kitty
  graphics protocol support for `foot` and Konsole (neither implements it; both are Sixel terminals, now
  detected and used instead); and the whole preview pane going blank after a Draw (writing the image bypassed
  prompt_toolkit's own cursor-position tracking, corrupting the next redraw — fixed by keeping it in sync).
- 2026-10-08: added `chafa` (plain ANSI/Unicode block-character art; needs no terminal graphics protocol at
  all) as a further preview fallback below Kitty/iTerm2/Sixel, and made the whole preview pane optional: with
  none of the four available (checked at startup, not guessed), the Molden/cube screen is just the
  file/settings pane at full width, with no empty space held open for a preview that cannot happen; Draw and
  Compare still work exactly the same, just always opening the plot in the system's image viewer.
- 2026-10-08: v04.06's bare `python newBRORSEN.v04.06.py file.molden` auto-detection (any-extension sniffing of
  every positional argument) is removed in v04.07; the viewer is reached only through `--ui` (the button, or
  `--ui job.molden`) or the explicit `python newBRORSEN.v04.07.py --view file.molden [file2]` (the older,
  non-`--ui` menu, unchanged).
- 2026-10-09: fixed a v04.06 bug (reported externally while preparing a pyscf_neo_cc author showcase, not a
  library issue): `%cube` / `%molden` crashed `export_wavefunction()` on a POSITRON stage with an odd or
  higher-spin positron count (`%n_pos 3 %pos_mult 4` and similar: the spin-orbital POSSO route), right after
  the SCF had already converged correctly (`ValueError: operands could not be broadcast together with shapes
  (2,48,48) (2,48)`). `mf.mo_coeff`/`mf.mo_occ` get an extra leading spin-channel axis in that case (shape
  `(2, nao, nmo)` / `(2, nmo)`), which `(c * occ) @ c.T` cannot broadcast; fixed by summing both channels
  into one combined density (new helper `density_from_mo`), matching every other density in the function. A
  second, related bug surfaced fixing the first: `molden.from_mo` (PySCF's own writer) does not accept that
  combined shape either, only a plain `(nao, nmo)` array per spin channel — fixed by writing one Alpha and
  one Beta block into the same file (as PySCF's own `molden.dump_scf` does for UHF), instead of a single
  call. Also fixed in the `--ui` input builder: `%pos_mult`/`%pos_spin` were a fixed `("1","3")`/`("0","2")`
  choice list (valid only for the closed-shell pair, `n_pos` up to 2) and were offered only when `%n_pos`
  was kept as exactly `2`; both now accept any integer and are offered whenever `%n_pos` is kept at all,
  so the `--ui` screen no longer flags this as an error nor hides the directive for `n_pos` >= 3. Regression
  test: `tests/test_v0405.py::test_molden_and_cube_three_positrons`, input
  `tests/inputs/export/files_three_positrons_quartet.inp`.
- 2026-10-09: two small `--ui` corrections (user request). `%conv_check` is a DF-vs-CONV cross-check, so it
  makes no sense once CONV is chosen — it is no longer offered then (same `fitting sets under CONV` rule
  as `%jkfit_aux_e` and friends). A missing basis-set option now gets a yellow `W`, once a sector has
  actually been chosen (nothing before that): `%basis` for any sector, `%pos_basis` for POSITRON,
  `%nuc_basis` and `%quantum_nuc` for PROTON — each has a working default, so this is a nudge, not an error.
  While its gray template line is still offered, `W` replaces the usual gutter character right there; once
  it is gone (typed over, or an opened file that never had it), the same `W` shows on the stage's `!` line
  instead. A continuation stage that inherits the option from the stage before does not trigger it.
  `%density_map_combine` (the new directive below) is now suggested only once `%density_map` is kept,
  instead of in every template regardless of whether density maps are involved. Also new: F3 (or vim
  `:done`) stops offering the current stage's template immediately, the same as saving would, without
  saving.
- 2026-10-13: the `--ui` input builder is split out of `newBRORSEN.v04.07.py` into its own program,
  `input_builder.v04.07.py` (user request). It drives the engine only through that script's command line
  (`--dry-run`, `--parse-text-json`, `--view`, a plain run) and, for the Molden/cube viewer, by loading its
  module namespace lazily with `runpy`; the engine script to use is Settings > **Engine path** (new row:
  `ui_engine_path`/`ui_default_engine_guess`), auto-detected when a matching-version
  `newBRORSEN.vNN.NN.py` sits next to the input builder. `tests/test_ui.py` now loads the two files
  separately (`W` for the UI, `E` for the engine) for its sync checks.
- 2026-10-13: **New input** now asks for the geometry first (type/paste/load-file, the same menu F2 opens
  later), skipped when one is already given another way (command line, a `.xyz`/`.zmat` argument, Open);
  the separate "New from geometry" button is gone, folded into this (user request; reverses the "geometry-
  first stays optional" call from 2026-10-08). `%basis`, `%memory` and `%nprocs` now start as real lines
  with their usual default instead of gray suggestions. `%jobname` is auto-generated
  (`formula_method_q<charge>_m<mult>_basis`, e.g. `H2O_HF_q0_m1_ccpvdz`) and stays current as the geometry,
  `!` method, charge/mult or `%basis` change, until it is edited by hand, after which it is left alone.
  A `%basis` that does not cover every element in the geometry (checked against the installed basis set's
  own data file) is now flagged yellow, naming `def2-tzvp`/`def2-qzvp` when one of those covers them
  instead. Section 2.1, decisions log D63.

**v04.08 (2026-10-14; development build, not released):** the v04.07 development build renamed (user's
decision; `newBRORSEN.v04.08.py`, `input_builder.v04.08.py`, deployed to `~/newBRORSEN_v04.08_deploy/wrapper/`
on both machines; v04.07 kept as its own file), plus:

- 2026-10-14 (engine, v04.08): new input checks, found on a user's multi-stage CHELPG input. A misspelled
  `%basis` (e.g. `aug-cc-pvt`) or one without functions for an element of the geometry is a validation error
  (it used to pass and stop the run at the first SCF). A second `%jobname` or a second geometry block in one
  stage is an error that names the likely missing `%next` (before, the later calculation silently replaced the
  earlier one). ZOPT with an all-number Z-matrix makes equal bonds variables (10.1).
- 2026-10-16 (engine, v04.08): `%scan bond|angle|dihedral <atoms>` or `%scan <zmat variable>` with
  'start end n' segments (one line or a block) or `[v1 v2 ...]` values; Cartesian angle and dihedral scans
  are new (section 12.1). The input builder reads the new block form. `%scan_print eda`: the energy
  components of every scan point in `<job>_scan_eda.txt`.
- 2026-10-16 (engine, v04.08): electronic / positronic energy partition of HF, MP2/CCSD correlation and
  (T) for POSITRON stages (section 10, energy components); the (T) terms by type; `%mp2_partition true`
  adds the DF-MP2 split to a CC stage.
- 2026-10-15 (engine, v04.08): `%pos_basis PSX-6Z` (9s7p6d5f4g3h2i; section 8.1); the input builder lists it.
- 2026-10-15 (engine, v04.08): Morse fit reliability warnings (R^2 < 0.995, fewer than 3 points in the
  well) and a curvature-based starting width (section 12.2).
- 2026-10-14 (engine, v04.08): `%pos_basis` and `%basis` take atoms (element symbols, atom numbers, `{...}`)
  and several bases on one line (section 8.2).

**v04.06 (2026-10-06; production release, signed off 2026-10-06):** v04.05 plus a file viewer:
`python newBRORSEN.v04.06.py file.molden [file2]` (or `.cube`) opens a Multiwfn-style menu that draws the plane
maps (density, ESP) with every option and computes CHELPG charges, with no calculation. Molden files reproduce the
calculation's own maps (maxima to 1e-3, the same ESP range, CHELPG q(Li) 0.794309 for e+ LiH against Multiwfn's
0.794308); cubes are interpolated trilinearly (0.5% electrons, 2.5% positron away from the nuclei at 80 points) and
give density maps only. (A `! PLOT` stage type tried earlier the same day was replaced by this menu.) Tests:
`tests/test_viewer.py`, inputs `tests/inputs/viewer/`.

- 2026-10-06: `%pos_basis PSX-5Z` (8s6p5d4f3g2h, built here with the PsX recipe; section 8.1); `make_input.py`
  offers 5Z (aug-cc-pV5Z + PSX-5Z).
- 2026-10-05: ZOPT no longer loops forever when the minimum lies past a variable's limit (angles 1-179 deg): a
  local sample beyond the limit was clipped back onto a known point and requested again without end (Na2H2,
  `results/Tests/ZOPT_bug_test`: 1100+ cached evaluations). Samples are now compared after clipping, a run that
  only finds known points stops ("stalled"), and a variable that ends at its limit gets a warning explaining that
  the minimum lies outside what the Z-matrix can describe.
- 2026-10-05: hybrid-ZOPT options (`%zopt_gradient_every`, `%zopt_final_gradient`) carried over into a stage whose
  method has no analytic gradient (e.g. DF-CCSD after HF) are switched off there with a warning instead of refusing
  the input; written in that stage itself, they are still an error.
- 2026-10-05: a stage without a geometry block that continues a Z-matrix stage keeps its Z-matrix, so
  `%opt_engine zopt` chains (e.g. ZOPT HF, then CCSD, then CCSD(T)) work; each continues from the previous
  stage's optimized variables (its `_final.zmat` after `--resume`). Continuing a geomeTRIC/WOPT optimization
  with ZOPT is refused with an explanation (the variables no longer describe the geometry).
- 2026-10-01: a list/block option written in a continuing stage (`%constraints`, `%pin`, `%pinnings`,
  `%equivalence(s)`, `%density_map`) replaces the inherited one; up to then it was appended
  (a repeated `%constraints` block doubled the constraint). Two stages with the same `%jobname` get a warning
  (the later overwrites the earlier's files); `! OPT FREQ` is exempt.
- 2026-09-30: two words of one `!` group (`! HF CCSD`, `! DF CONV`, `! OPT SCAN`) are an error; `! OPT FREQ` works
  as in ORCA (an OPT stage, then a FREQ stage at its result).
- ELECTRONIC `OPT` runs geomeTRIC (section 10), with a WOPT fallback. `%opt_engine WOPT` keeps the built-in
  engine.
- Electronic CCSD gradients are analytic through `nuc_grad_method()`; up to v04.04 they were silently
  numerical.
- DF-CC gradients are numerical: exact for the DF energy, where PySCF's formula is 3-5e-5 Ha/Bohr off.
- `GRADIENT` jobs never start an optimization (electronic CC and positron HF did when the analytic gradient
  was missing).
- PySCF's DZ JK-fit sets are replaced by `aug-cc-pvtz-jkfit` for Ga-Kr (section 7.1).
- `--resume`, automatic after a SLURM requeue: stage records (written in SLURM jobs only, or with
  `--stage-records`), skipping of finished stages, and restart of an
  interrupted optimization from its checkpoint (section 2).
- An unrecognized directive is an error (it was a warning, so a misspelled setting was silently skipped).
- `%molden` / `%cube`: Molden and cube files of the HF and CCSD wavefunctions, electrons and positrons (section 13).
- `%chelpg` / `%density_map`: CHELPG charges in the `.out` and density maps in the job folder, computed by the
  wrapper with no extra program (section 13).
- Energy components at the final geometry (HF, MP2 or CCSD, (T), CCSD(T)), for the DF and the CONV energy,
  in the log and in `<job>_summary.txt` (section 10). Up to v04.04 a CCSD(T) optimization never showed its
  CCSD energy.
- Tests: `tests/test_v0405.py`.


**v04.04 (2026-09-24 to 2026-09-26; production release, signed off 2026-09-26):** optimization results are written before the `%conv_check`
single point in all three engines (geomeTRIC, WOPT, ZOPT); the summary is rewritten with the CONV
energy when the check succeeds. Test: `tests/test_conv_check_safety.py` (kills the process inside the
check; v04.03 lost the files, v04.04 keeps them).
Also in v04.04 (2026-09-26): the two-positron `%df_aux` set per element (RI for H-Ne, JK-fit for heavier
atoms) and the **pre-flight check** of the two-positron DF gradient (exit code 3; section 7.1; tests
`tests/test_pair_aux.py`); a warning for PySCF's DZ JK-fit sets for Ga-Kr (no g functions: Br- 674 mHa).
**Constrained NEO** in the PROTON sector (`%cneo`, `%cneo_variant`, the
words `CNEO` and `CDFT`; CNEO-HF, CNEO-DFT, CNEO-MP2, CNEO-CCSD, CNEO-CCSD(T); section 7.3; tests
`tests/test_cneo.py`), with PROTON fixes: analytic HF/DFT gradients and geomeTRIC (they fell back to
numerical WOPT), `%frozen` honoured everywhere, `%epc none` (an unknown `%epc` silently became 17-2 and is
now an error). Also in v04.04 (2026-09-25; superseded on 2026-09-26 by the per-element choice above): the two-positron `%df_aux` `auxbasis_e` defaulted to the matched JK-fit set
and an RI set there was replaced (section 7.1): with an RI set the two-positron CC reference was misfitted
(Br2 2e+ 4.9 Ha too low; Cl2 2e+ 4.9 mHa). Added later on 2026-09-24, in the same file (version number
kept at the user's request): **MP2** in
all three sectors (section 7.5; tests `tests/test_mp2.py`); hybrid ZOPT with optional analytic gradients (`%zopt_gradient_every`, `%zopt_final_gradient`; section 10.1;
`tests/test_zopt_hybrid.py`); positron-HF optimizations say that they run
WOPT with numerical gradients (validation warning, summary engine; section 10; `tests/test_engine_labels.py`);
**particle
masses** (`%pos_mass`, `%quantum_site`, `%nuc_mass`; the library's protonic nuclear basis sets in both
sectors, scaled to the mass; an unknown `%nuc_basis` is a validation error; section 7.4; tests
`tests/test_masses.py`); **ghost atoms** (section 6.3; tests `tests/test_ghost_atoms.py`), fixing three
bugs present up to v04.03: `%frozen auto` counted the core of ghost atoms, the `ghost_X` spelling was
read as the element `_x`, and after a geomeTRIC optimization the atom labels were rebuilt from PySCF's
pure symbols, which turned the NEO isotope marks `H*` / `H+` into plain H.

**v04.03 (2026-09-23, unreleased):** measured fitting-set defaults: mean field = the JK-fit set matched
to `%basis` (`aug-cc-pvtz-jkfit`), also replacing a moved RI set; electronic DF correlation = the matched
RI set; `%jkfit_aux_e auto`; the summary shows the fitting sets. Tests: `tests/test_matched_fitting.py`.

**v04.02 (2026-09-23, unreleased):** fitting-set roles for pyscf_neo_cc 2026-09-23: an RI set
never reaches the mean field (moved from `%jkfit_aux_e` to the correlation role with a warning);
CONV stages and `%conv_check` use an exact mean field; on the new library the DF recipe is given to the
mean field (no exact four-index e-p tensor) and the bug-#2 warning is shown only on the old library; template fixed (it used `aug-cc-pvtz-ri` as
`%jkfit_aux_e`). Tests: `tests/test_fitting_roles.py`, `tests/test_new_library.py`; the v03.17 numerical regression tests were
retired (`tests_retired/`). Everything else is v04.01.

**v04.01 (2026-09-22, unreleased):** gradient-free Z-matrix optimizer (`%opt_engine zopt`, section 10.1)
with `%zopt_*` directives and restart; comma-separated Z-matrix rows; Z-matrix dummy atoms
(`X`, `Du`); error for atoms closer than 0.5 A. Everything else is v04.00.

**2026-09-22:** `PSX-DZ`/`PSX-TZ` are the openLOWDIN sets (DZ now 5s3p2d: 10 shells, 24 functions); `PSX-QZ` added.

**2026-09-22:** scans: `! SCAN` with `%scan_fit` (points only, Morse fit or harmonic fit: xe, k, omega_e, anharmonicity check); Morse fits also report omega_e and omega_e x_e; fitted geometries are passed on for every scan mode.

**v04.00 (2026-09-21, unreleased)** - built on `newBRORSEN.v03.17_merged.py` (md5 fbd3c413...).

Added: twoposBrorsen/v5 input syntax (positional `!`, `%threads`, `%spin`, `%e_basis`,
`%verbose`, `%optimizer`, `%key=value`, `%pos_basis` literals, `%pinnings`/`%equivalences`
blocks, `%scan` blocks); Z-matrix input with variables; `bond`/`translate`/`sym_zmat`/`fragment`
scans for any molecule; per-stage `.scan` tables; triplet positron pairs; `%df_aux`;
`%df_mem_cap`; `%block_size_a`; `PSX-QZ`; `%opt_max_cycles`;
`%conv_check`; SCF ARH/Newton fallback; `--dry-run`; full validation pass with errors and
warnings; version in summaries.

Fixed (present in v03.17 and/or twopos): pinned atoms drifting; wrong dihedral gradient and
sign; ELECTRONIC CCSD(T) gradient inconsistent with its energy; bare `*` not closing an xyz
block; trailing integer constraint target read as an index; geomeTRIC constraint targets
ignored; memory of the seed-geometry objects
held during the final single point; one-line scan with < 4 points aborting; unknown `!`
tokens silently ignored.

Unchanged by design: every computational path of v03.17 (tests prove identical numbers when
both wrappers get the same positron basis).

## Explicit auxiliary conditioning

The opt-in POSITRON DF fixed/pivoted-shell policy and required restart sidecar
are documented in [AUXILIARY_CONDITIONING.md](AUXILIARY_CONDITIONING.md).
The default remains unconditioned.
# Public positron numerical frequencies

The integrated wrapper uses the public numerical Hessian driver for supported
one/two-positron frequencies. See [NUMERICAL_HESSIAN.md](NUMERICAL_HESSIAN.md)
for raw diagnostics, native checkpoint resumption, solve counts and the explicit
triples-blocking exception. This supplements the historical frequency section.


## v04.09: Kurt Brorsen's 2026-10-08 integration (pyscf_neo_cc 0.1.0.post20261008)

Merged 2026-10-20 from the wrapper integration Kurt Brorsen sent with the library. It was written against the
v04.05 he received (2026-09-29). It was merged three-way into v04.09 with that file as base. Each feature is
described in its own note in `kurt_2026-10-08/` (copied from the package):

| Feature | Note |
|---|---|
| Two-positron DF reference: `reference_model='scf'` for n_pos = 2 (the library default; 'legacy' only reproduces old numbers) | `PAIR_REFERENCE_POLICY.md`, `pair_reference_model.md` |
| Explicit fixed-shell conditioning of dependent fitting sets (`%aux_conditioning`) | `AUXILIARY_CONDITIONING.md` |
| Gradient / optimization / frequency admission checks before correlation work | `GRADIENT_ADMISSION.md`, `GRADIENT_CAPABILITIES.md`, `NEO_GRADIENT_ADMISSION.md` |
| Analytic POSHF forces; geomeTRIC POSHF optimization; POSHF SCF recovery, also in scans | `POSHF_WORKFLOWS.md` |
| Native numerical Hessians; analytic single-positron CCSD Hessian (CONV, single-centre positron basis, zero field) | `NUMERICAL_HESSIAN.md`, `ANALYTIC_HESSIAN.md` |
| Digest-keyed result, Hessian, ZOPT and --resume caches (source, surface and basis-file identity) | `CACHE_IDENTITY.md`, `OPTIMIZER_SEED.md` |
| Declared-density ESP / CHELPG via the library's `density_model`; positron charge sign | `CHELPG_REFERENCE_CONVENTION.md` |
| Library release notes and stated limits | `RELEASE_2026-10-08.md` |

Measured effect of the pair reference (F2 2e+, R = 3.0 A, aug-cc-pVDZ / PSX-DZ, DF): POSHF identical;
CCSD and CCSD(T) both 3.12 mHa higher than with the 2026-09-23 library (CCSD(T) -199.772629875848 ->
-199.769510919761 Ha). Two-positron numbers from v04.08 and v04.09 are therefore not interchangeable.

**`%pair_reference scf | legacy`** (v04.09, default `scf`): the reference model of the two-positron DF classes
(POSDFCCSD, POSSODFCCSD, POSDFMP2). `scf` is the library's corrected reference (Kurt's recommendation). `legacy`
exists only to reproduce numbers from v04.08 and older for comparison. Checked 2026-10-20: F2 2e+ (R = 3.0 A,
aug-cc-pVDZ / PSX-DZ) gives the v04.08 CCSD and CCSD(T) to all 12 decimals (-199.772629875848 Ha). For Cl, Br and heavier atoms, v04.08 also used JK-fit correlation sets where v04.09 uses matched RI, so
reproducing such a number also needs that set given in `%df_aux auxbasis_e`. The run summary prints the model in
use ("Pair reference:"). Ignored, with a warning, outside two-positron DF stages.

**Library check (v04.09).** The wrapper stops at start-up, before reading the input, unless pyscf_neo_cc is
0.1.0.post20261008 or newer and has the positron sector, DF-MP2, POSHF forces and POSHF.recover_scf. The
error names the env to use. The 2026-09-23 library's feature switches were removed with this.

**Fixed in v04.09:** an `H+` / `H*` / `H#` quantum nucleus with a `%nuc_mass` value had its nuclear basis scaled
twice: once by the library for the isotope symbol, once by the wrapper relative to the proton. The wrapper now
applies only the remaining factor sqrt(m / m_isotope). Checked: two-positron MP2 OPT and GRADIENT use the
library's analytic POSDFMP2 gradient. LiH 2e+ (6-31G / PSX-DZ, cc-pvdz-ri): analytic -0.0157358919 vs. central
difference -0.0157359633 Ha/Bohr.

Kept from v04.06-v04.08 on top of Kurt's changes: stage inheritance and Z-matrix continuation, %scan_print eda and
the energy partition, basis groups / ranges, file paths with spaces, the ghost-atom basis key, the two-positron
CONV CCSD retry, the comment-insensitive --resume signature (now also keyed to the code version).
