"""Loader for BioTek/Agilent Gen5 plate-reader exports (``.txt`` and ``.xlsx``).

By default, a Gen5 export contains one or more **plate blocks**, each beginning
with a ``Software Version`` line. Within a block the layout has three parts:

1. **Header** — tab-separated ``key<TAB>value`` lines identifying the plate and
   run information (``Plate Number``, ``Date``, ``Time``, ``Reader Type``, ``Reader Serial
   Number``, …).
2. **``Procedure Details``** — the instrument program: the ``Plate Type`` plus
   one or more directives, such as ``Read`` and ``Shake``. Each read declares a
   modality (Fluorescence, Absorbance, or Luminescence; Alpha and Image are
   recognized but unsupported) + type (Endpoint, Spectrum, Area Scan) and,
   indented beneath it, the optics for that read — ``Filter Set`` blocks for
   fluorescence/luminescence, a ``Wavelengths:`` list for absorbance, or
   ``Start``/``Stop``/``Step`` for spectra. Reads may be nested inside
   ``Start Kinetic … End Kinetic`` to mark them kinetic.
3. **Data sections** — blank-line-delimited segments holding the actual
   measurements, in one of two shapes: a **grid** (a plate matrix, rows x plate
   columns, with the read identifier in a trailing column) or a **table**
   (leading ``Time``/``Well``/``Wavelength`` column, one row per timepoint or
   well). Interspersed are single-line markers (read labels, ``Actual
   Temperature``) and non-measurement sections: ``Layout`` grids are skipped by
   name, and curve fits and the like are dropped when they parse to no usable
   data column.

**Two-phase parse.** We first parse the procedure details and record the information (``_parse_procedure_section`` →
``_build_read_metadata``), then read the data sections, then match read labels in the data section
with corresponding metadata in the procedure details (``_match_read`` → ``_get_closest_match``) and attach
the matched metadata (modality, Ex/Em wavelengths, optics, geometry, etc.) onto each data segment's frame.
The loader returns one ``PlateReaderData`` per data format found (kinetic / spectrum / endpoint), not a single merged frame.

In the data section data are annotated with a read label. The most detailed read label takes the form
``<Read Name>:<Ex wv>/<Ex bw>,<Em wv>/<Em bw>`` for fluorescence, ``<Read Name>:<Em wv>/<Em bw>`` (or
``<Read Name>:Lum``) for luminescence, or ``<Read Name>:<Wv>`` for absorbance (and ``<Read Name>:Spectrum``,
``EM Spectrum``, or ``EX Spectrum`` for a spectrum). It can have these modifications:
- If there is only one read directive in the procedure, the read name will be omitted in the data section
- If the read is not named by the user in Gen 5, by default the read is named ``Read N`` where ``N`` is the read's
  1-based position among all read directives in the procedure (the counter increments for every read, named or not)
- In the data section, the bandwidth seems to be omitted for monochromator reads (verified for Cytation 3 with both optics)
- The read name may be contained by a stat wrapper (``Max V [Read 2:450,490]``), or have a polarization
channel tag (``... [Perpendicular]`` or ``... [Parallel]``). 
The procedure section separately records the read name and wavelengths. 

To match read data with procedural read metadata, we first construct detailed candidate read labels from the metadata.
Then bare read name strings, both true data labels and candidates, are parsed into ``ReadKey`` objects, which separate
out these components. We perform matching in a tiered manner: exact ``ReadKey`` equality, then wavelength+tag matching (which
disambiguate the fluorescence-polarization channel cases), which falls back to a
permissive substring-containment search.

**Optics caveat.** Whether a read used a filter or a monochromator is not stated
outright; it is inferred — first, at parse time from the presence/absence of the
``Optics``/``Mirror`` fields, then overridden per-axis (excitation/emission) from the data-grid label
(a ``/`` bandwidth in the wavelength token means filter; ``_apply_label_optics``). 
The inference can be wrong; the loader logs a warning to that effect.

**``.xlsx`` support.** Gen5 ``.xlsx`` exports are the same logical document laid
out in spreadsheet cells. Rather than write a second parser, the workbook is
rendered back into the exact lines its ``.txt`` export would have contained
(``_xlsx_rows_to_lines``) and fed through the identical block/segment machinery.
The Gen5 ``.xlsx`` writer breaks several conventions the line parser relies on —
every data table/grid is shifted one column to the right, there is no blank line
after ``Procedure Details``, the endpoint grid is detached from its ``Results``
label, and numbers/dates are typed cells rather than text — so the cell→line
conversion corrects for each (see ``_format_xlsx_cell`` and
``_xlsx_rows_to_lines``).
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import re
from collections.abc import Collection
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from ..base import PlateReaderLoader, plate_loader
from ..result import PlateReaderResult, PlateReaderData, PlateReaderDataFormat
from ..constants import (
    MetadataColumn,
    PlateReaderDataColumn,
    ReadModality,
    REQUIRED_DATA_COLUMNS,
    REQUIRED_METADATA_COLUMNS,
    REQUIRED_READ_COLUMNS,
)

log = logging.getLogger(__name__)

_INVALID = {"OVRFLW", "?????", "MISSED"}
MONOCHROMATOR = "monochromator"
FILTER = "filter"

# warn the user if Reader Serial Number is one of these
_READER_NA = {"", "n/a", "Unknown"}

# Minimum header keys required to identify a plate block.
_REQUIRED_HEADER_KEYS = {
    "Plate Number",
    "Date",
    "Time",
    "Reader Type",
    "Reader Serial Number",
}

# Maps keys in Gen5 default header to canonical names
_RENAME_MAP = {
    'Reader Type': MetadataColumn.READER_TYPE,
    'Reader Serial Number': MetadataColumn.READER_ID,
    'Plate Number': MetadataColumn.PLATE_ID,
}

# Action directives that can appear in the procedure block:
_ACTIONS = ("Read\t","Start Kinetic\t","Dispense\t","End Kinetic", "Set Temperature", "Shake\t")
_SUPPORTED_READ_MODES = set(ReadModality)
_SUPPORTED_READ_TYPES = {"Endpoint", "Spectrum", "Area Scan"}

# For now, skip metadata parsing for these types:
_NOT_SUPPORTED_READ_MODES = {"Alpha", "Image"}

# Possible first columns for data tables
_TABLE_START_COLS = ("Time\t", "Well\t", "Wavelength\t", "X Read #\t", "Y Read #\t")

# Procedural metadata that should be extracted for spectrum reads:
_SPECTRUM_READ_DATA = {'Emission Start', 'Excitation Start', 'Start',
                       'Stop', 'Step', 'Optics', 'Gain', 'Mirror',
                       'Fixed Excitation', 'Fixed Emission'}


@plate_loader(name="biotek", extensions=[".txt", ".xlsx"])
class BiotekLoader(PlateReaderLoader):

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str, keep_diagnostic_metadata: bool = False) -> PlateReaderResult:
        """Load a Gen5 ``.txt``/``.xlsx`` export into a ``PlateReaderResult``.

        Reads the file (rendering ``.xlsx`` to its equivalent ``.txt`` lines
        first), splits it into plate blocks (starting with ``Software Version`` lines),
        and parses each block independently via ``_parse_block``. The per-block
        results are flattened into one ``PlateReaderData`` list, with a parallel
        ``_plates`` index recording which plate block each came from and a per-plate
        metadata frame.

        Args:
            path: Path to the Gen5 export (``.txt`` or ``.xlsx``).
            keep_diagnostic_metadata: Retain internal bookkeeping columns
                (``inferred_label``, ``supported_mode``, ``read_type``, …) on the
                output instead of dropping them — useful when debugging read
                reconciliation.

        Returns:
            A ``PlateReaderResult`` holding one ``PlateReaderData`` per parsed
            data format, the per-plate metadata frame, and the source file path.

        Raises:
            ValueError: If no plate blocks are found (typically a non-Gen5 file
                or a non-UTF-8 encoding).
        """
        path = Path(path)
        # Optics source (filter vs monochromator) is inferred from the Gen5
        # export rather than read directly, and may be incorrect.
        log.info("BioTek optics source (filter/monochromator) is inferred and may be incorrect.")
        if path.suffix.lower() == ".xlsx":
            lines = cls._read_workbook_lines(path)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()

        block_starts = [
            i for i, line in enumerate(lines)
            if line.startswith("Software Version\t")
        ]
        if not block_starts:
            raise ValueError("File does not appear to be formatted as a default Gen5 export type; could not parse. Check encoding type is UTF-8.")
        log.info(f"Found {len(block_starts)} plate blocks")

        results = []
        plates = []
        plate_meta_rows = []
        plate_idx = 0
        for idx, start in enumerate(block_starts):
            log.info(f"Parsing plate block {idx+1}")
            end = block_starts[idx + 1] if idx + 1 < len(block_starts) else len(lines)
            block_results, plate_meta = cls._parse_block(lines[start:end], keep_diagnostic_metadata=keep_diagnostic_metadata)
            if block_results:
                plates.extend([plate_idx] * len(block_results))
                plate_meta_rows.append(plate_meta)
                plate_idx += 1
            results.extend(block_results)

        plate_meta_df = pd.DataFrame(plate_meta_rows)
        if len(plate_meta_df) > 0 and len(plate_meta_df[MetadataColumn.PLATE_ID].unique()) < len(plate_meta_df):
            log.warning("Suspected duplicate plate(s) detected in the data file.")
        return PlateReaderResult(
            data=results,
            _plates=plates,
            metadata=plate_meta_df,
            source_file=path,
        )

    @classmethod
    def _read_workbook_lines(cls, path: Path) -> list[str]:
        """Render a Gen5 .xlsx export to the lines its .txt export would have.

        Concatenates the per-worksheet conversions so the rest of ``load`` can
        treat the workbook exactly like a ``.txt`` file. Caveat: kinetic times
        at or beyond 24 h are mis-rendered by openpyxl's 1900-epoch handling —
        not hit by current files (data ends 23:12)."""
        from openpyxl import load_workbook

        wb = load_workbook(path, data_only=True)
        lines: list[str] = []
        for ws in wb.worksheets:
            lines.extend(_xlsx_rows_to_lines(list(ws.iter_rows())))
        return lines

    # ------------------------------------------------------------------
    # Block orchestration
    # ------------------------------------------------------------------

    @classmethod
    def _parse_block(cls, lines: list[str], keep_diagnostic_metadata: bool = False) -> tuple[list[PlateReaderData], dict]:
        """
        Parse a block corresponding to data from one plate.

        Splits the block into blank-line-delimited segments, classifies each,
        and dispatches to the appropriate parser. Returns ``(results,
        plate_meta)`` where ``results`` is a list of ``PlateReaderData`` — one
        per data format found (kinetic, spectrum, endpoint).
        """

        segments = _split_segments(lines)
        if not segments:
            log.warning("Plate block is empty")
            return [], {}

        # Look for the "Procedure Details header"
        proc_marker_idx = next(
            (i for i, seg in enumerate(segments) if seg[0].strip() == "Procedure Details"),
            None,
        )
        if proc_marker_idx is None:
            log.error("No 'Procedure Details' block found.")
            return [], {}

        # Parse header (plate/experiment information)
        metadata = cls._build_plate_metadata(segments[:proc_marker_idx])
        if metadata is None:
            return [], {}

        # Read the procedure details section and extract read information
        proc_seg_idx = proc_marker_idx + 1
        proc_meta = {}
        if proc_seg_idx < len(segments):
            # Returns {Plate Type, "reads": [...]} where each read carries its
            # parsed read information.
            proc_meta = cls._parse_procedure_section(segments[proc_seg_idx])
        else:
            log.warning("No procedure content found after 'Procedure Details' marker.")

        log.debug(
            "Plate %s  |  %s %s  |  %s",
            metadata.get(MetadataColumn.PLATE_ID, "?"),
            metadata.get("Date", "?"),
            metadata.get("Time", "?"),
            metadata.get(MetadataColumn.READER_TYPE, "?"),
        )
        log.info("Read metadata section, found %d read block(s).", len(set(x[PlateReaderDataColumn.READ_NAME] for x in proc_meta['reads'])))

        metadata[MetadataColumn.PLATE_TYPE] = proc_meta[MetadataColumn.PLATE_TYPE]

        reference_read_map, read_record_df = cls._build_read_metadata(proc_meta)

        # Read data sections
        data_segments = segments[proc_seg_idx + 1:]
        log.debug(f"Parsing {len(data_segments)} data segment(s).")

        current_label = None

        results = []
        successes = 0
        unreconciled = 0
        for i, seg in enumerate(data_segments):
            metadata_read_map = {}
            seg_type = None
            # Classify segment type
            first = seg[0].strip()  # first line of block
            result = None
            if len(seg) == 1:  # it's a label or a temperature block
                if first.startswith("Actual Temperature"):
                    # Temperature marker — per-reading temperature comes from the
                    # data table's own column, so nothing to capture here.
                    log.debug(f"Data segment {i+1}: actual temperature")
                    current_label = None
                else:  # it's a label
                    log.debug(f"Data segment {i+1}: label")
                    current_label = first
            else: # it's either a data grid or a table
                # Try only to read tables corresponding to data we are interested in
                # So, skip "Layout" or "Conc" blocks, only read "Results" or tables line-separated from a read label
                matched_name = None
                if seg[1].startswith("\t"): # it's probably a data grid; doesn't always start with the word "Results"!
                    if seg[0] == "Layout":
                        log.debug(f"Data segment {i+1}: layout grid, skipping")
                        continue
                    if _looks_like_rowwise_results_table(seg):
                        raise ValueError(
                            "Unsupported Gen5 tabular (row-wise) results layout; "
                            "re-export as a plate matrix with column headers."
                        )
                    log.debug(f"Data segment {i+1}: data grid")
                    result = parse_data_grid(seg[1:])

                    if result is not None and PlateReaderDataColumn.DATA in result.columns and len(result) > 0:
                        if seg[0] == "Results":
                            # Should have a "Read" column, look at reads
                            if PlateReaderDataColumn.READ not in result.columns:
                                log.error(f"No {PlateReaderDataColumn.READ} column found in data grid; cannot assign read metadata.")
                            else:
                                unique_read_names = [str(x) for x in result[PlateReaderDataColumn.READ].unique()]
                                for read_to_match in unique_read_names:
                                    cls._match_read(read_to_match, reference_read_map, metadata_read_map)
                        else:  # check if it's a read name
                            cls._match_read(seg[0], reference_read_map, metadata_read_map)
                    else:
                        log.warning("Parsed data grid did not contain any data.")
                        current_label = None
                        continue
                    current_label = None
                else:
                    # Sometimes, data tables have a label header with sample IDs, skip it:
                    if first.startswith("Well ID\t") and len(seg) > 1:
                        seg = seg[1:]
                        first = seg[0].strip()
                    if first.startswith(_TABLE_START_COLS):  # read data table
                        # `current_label` should tell us the read type if this isn't a results grid.
                        # A "Well\t"-led table is a results table whose first column already holds the
                        # read names (incl. stat wrappers like "Corrected [A600:600]"; see the
                        # od600-growth-rescue fixture), matched per-row below — so it is not
                        # associated with current_label.
                        if not first.startswith("Well\t"):
                            matched_name = cls._match_read(current_label, reference_read_map, metadata_read_map) if current_label is not None else None
                        if first.startswith("Time\t"):
                            seg_type = "kinetic"
                            log.debug(f"Data segment {i+1}: kinetic block with label {current_label}")
                            if matched_name is not None and not reference_read_map[matched_name]['kinetic']:
                                log.warning(f"Data segment {i+1} with label {current_label} was not expected to be kinetic according to the metadata.")
                        elif first.startswith("Wavelength\t"):
                            seg_type = "spectrum"
                            log.debug(f"Data segment {i+1}: spectrum block with label {current_label}")
                            # Validation:
                            if current_label is not None and current_label.rpartition(":")[-1].find("Spectrum") == -1:
                                log.warning(f"Block type is a spectrum but label {current_label} does not look like a spectrum label.")
                        elif first.startswith("X Read #\t") or first.startswith("Y Read #\t"):
                            seg_type = "area_scan"
                            log.debug(f"Data segment {i+1}: area scan block with label {current_label}")
                        elif first.startswith("Well\t"):
                            seg_type = "results_table"
                            log.debug(f"Data segment {i+1}: results table")
                        else: # shouldn't be able to get here?
                            log.warning(f"Data segment {i+1}: unknown type, skipping.")
                            seg_type = "unknown"
                            continue
                        result = parse_data_table(seg)
                        if result is not None and PlateReaderDataColumn.DATA in result.columns and len(result) > 0:
                            if seg_type == "results_table":
                                # match reads
                                unique_read_names = [x for x in result[PlateReaderDataColumn.READ].unique()]
                                for read_to_match in unique_read_names:
                                    cls._match_read(read_to_match, reference_read_map, metadata_read_map)
                            else:
                                result = result.assign(Read=current_label)
                                if matched_name is not None:
                                    metadata_read_map[current_label] = matched_name
                                else:
                                    metadata_read_map[current_label] = current_label
                        else:
                            log.warning("Parsed data table did not contain any data.")
                            current_label = None
                            continue
                    current_label = None
                # skip curve fits, etc.

                # Generate results
                if result is not None:
                    successes += 1
                    if len(metadata_read_map) == 0:
                        unreconciled += 1
                    results.append(cls._assemble_result(
                        result,
                        metadata_read_map,
                        read_record_df,
                        metadata,
                        seg_type,
                        keep_diagnostic_metadata,
                    ))

        log.info(f"Parsed {successes} segment(s) successfully.")
        log.info(f"Found {unreconciled} segment(s) that could not be associated with existing read data.")

        plate_meta = {
            MetadataColumn.PLATE_ID: metadata.get(MetadataColumn.PLATE_ID),
            MetadataColumn.DATE: metadata.get(MetadataColumn.DATE),
        }
        return results, plate_meta

    # ------------------------------------------------------------------
    # Plate & procedure metadata
    # ------------------------------------------------------------------

    @classmethod
    def _build_plate_metadata(cls, header_segments: list[list[str]]) -> dict | None:
        """Build the plate metadata dict from the header segments.

        Merges the header key-value segments, verifies the required keys are
        present, renames to the canonical schema, and parses the start
        datetime. Returns ``None`` if a required header key is missing (the
        caller should abort the block).
        """
        metadata: dict = {}
        for seg in header_segments:
            metadata.update(cls._parse_header(seg))

        # Check that the minimum required keys for identification purposes are present
        missing = _REQUIRED_HEADER_KEYS - metadata.keys()
        if missing:
            log.error("Missing required header keys: %s", missing)
            return None

        metadata = cls._rename_keys(metadata)
        if metadata[MetadataColumn.READER_ID] in _READER_NA:
            log.warning(f"Reader Serial Number in metadata '{metadata[MetadataColumn.READER_ID]}' does not allow for unique identification.")

        try:
            start_dt = pd.to_datetime(
                            f"{metadata.get('Date')} {metadata.get('Time')}", format="%m/%d/%Y %I:%M:%S %p"
                        )
        except ValueError as e:
            log.warning(f"Date was improperly formatted: {e}\n\n Trying auto inference")
            try:
                start_dt = pd.to_datetime(
                            f"{metadata.get('Date')} {metadata.get('Time')}", format='mixed')
                log.debug(f"Inferred date as: {start_dt}")
            except ValueError:
                log.warning(f"Date still improperly formatted: {e}")
                start_dt = pd.NaT

        metadata.pop('Date')
        metadata.pop('Time')
        metadata[MetadataColumn.DATE] = start_dt
        return metadata

    @classmethod
    def _build_read_metadata(cls, proc_meta: dict) -> tuple[dict, pd.DataFrame]:
        """Build the read-metadata structures from the parsed procedure section.

        Returns ``(reference_read_map, read_record_df)`` where
        ``reference_read_map`` is the read-matching set: a dict mapping
        ``"<read name>:<label>"`` keys (the supported reads only) to their read
        info dicts. Its keys are the candidates a data read is matched against
        (see ``_get_closest_match``); its values are consulted only for the
        post-match ``['kinetic']`` check. ``read_record_df`` is the attach-table
        — a DataFrame indexed by match key — each row is
        the per-read metadata record (invalid sentinels nulled, all-NaN
        columns dropped) ready for direct attachment to result rows without
        a string-join step.  Bare-name rows (one per ``read_name``, covering
        columns constant across all wavelengths of that read) are appended so
        that wavelength-free data reads can still match.
        Raises if no supported reads were parsed.
        """
        reference_read_map = {
            f"{read_data[PlateReaderDataColumn.READ_NAME]}:{read_data['label']}": read_data
            for read_data in proc_meta['reads']
            if read_data.get('supported_mode') and read_data.get('supported_type')
        }
        if len(reference_read_map) == 0:
            raise ValueError("No metadata was parsed; check if reads were of unsupported types.")
        metadata_df = (
            pd.DataFrame(proc_meta['reads'])
            .map(lambda x: np.nan if x in _INVALID else x)
            .dropna(axis=1, how='all')
        )
        metadata_df['inferred_label'] = (
            metadata_df[PlateReaderDataColumn.READ_NAME] + ':' + metadata_df['label']
        )
        metadata_df = metadata_df.drop(columns=['label', 'supported_mode', 'supported_type'])

        # Bare-name fallback rows: columns constant within each read_name group,
        # keyed by the bare read_name so wavelength-free data reads can still
        # match without a separate group_meta concat in every data segment.
        nunique = metadata_df.groupby(PlateReaderDataColumn.READ_NAME).nunique()
        shared_cols = nunique.columns[(nunique <= 1).all()].tolist()
        group_meta = (
            metadata_df
            .groupby(PlateReaderDataColumn.READ_NAME)[shared_cols]
            .first()
            .reset_index()
        )
        group_meta['inferred_label'] = group_meta[PlateReaderDataColumn.READ_NAME]
        read_record_df = pd.concat([metadata_df, group_meta], ignore_index=True)
        return reference_read_map, read_record_df

    @classmethod
    def _parse_header(cls, lines: list[str]) -> dict:
        """
        Parse plate/experiment metadata block. Returns a dictionary of key-value metadata pairs.
        """
        metadata: dict = {}
        for line in lines:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                key = parts[0].rstrip(":")
                metadata[key] = parts[1].strip()
        return metadata

    @classmethod
    def _rename_keys(cls, metadata: dict) -> dict:
        """Canonicalize header keys via _RENAME_MAP; pass others through."""
        return {_RENAME_MAP.get(k, k): v for k, v in metadata.items()}

    @classmethod
    def _parse_procedure_section(cls, lines: list[str]) -> dict:
        """Extract Plate Type and the reads list from the Procedure Details.

        Returns ``{Plate Type: <str>, "reads": [<read record>, ...]}``.

        This is a **line-driven state machine**, not a block parser: the
        procedure section has no blank-line structure, so reads are delimited
        only by markers. A read opens on a ``Read<TAB>…`` line and stays open —
        accumulating the indented detail lines that follow (mode/type, filter
        sets, shared key/value metadata) — until the next ``Read``, the next
        top-level action (``Start Kinetic``, ``Dispense``, ``Shake``, …; see
        ``_ACTIONS``), or end of section. The closing step calls the nested
        ``_flush_read`` helper, which turns the accumulated state into one or
        more flat read records (one per filter set / wavelength / spectrum axis)
        and appends them to ``result["reads"]``.

        The mutable state carried across lines (``read_name``, ``read_modality``,
        ``filter_sets``, ``is_kinetic``, ``expecting_mode``, the
        ``supported_*`` flags, …) is declared up front and reset by
        ``_flush_read``. ``is_kinetic`` is the one flag that persists *across*
        reads — it is toggled by the ``Start Kinetic``/``End Kinetic`` markers
        and stamped onto every read flushed while inside that window.
        """
        result: dict = {MetadataColumn.PLATE_TYPE: None, "reads": []}

        # Iterate through lines of the procedure section, and add reads in the order they are
        # encountered. Reads start with "Read\t" and end at the next read, the next directive,
        # or the end of the section

        supported_mode: bool = False
        supported_type: bool = False
        read_name: str | None = None
        read_modality: str | None = None
        read_type: str | None = None
        read_shared_meta: dict = {}
        filter_sets: list[dict] = [] # reads --> filter_sets for fluorescence, luminscence; wavelengths for absorbance
        expecting_mode: bool = False
        read_number: int = 0

        is_kinetic: bool = False

        def _parse_kv_line(line: str) -> dict:
            """Parse 'Key: Value,  Key2: Value2' into {'Key': 'Value', 'Key2': 'Value2'}.

            Segments without a colon are stored as True (e.g. 'Extended Dynamic Range').
            Exception: a purely numeric segment (e.g. '325' in 'Wavelengths: 280, 325')
            is appended to the previous key's value rather than stored as a boolean.
            """
            result: dict = {}
            last_key: str | None = None
            for segment in line.split(","):
                segment = segment.strip()
                if not segment:
                    continue
                if ":" in segment:
                    key, _, val = segment.partition(":")
                    last_key = key.strip()
                    if last_key == "Gain":
                        last_key = PlateReaderDataColumn.GAIN
                    elif last_key == "Read Height":
                        last_key = PlateReaderDataColumn.READ_HEIGHT

                    result[last_key] = val.strip()
                elif last_key is not None and segment.replace("/", "").isdigit():
                    result[last_key] += ", " + segment
                else:
                    if segment == "Full Plate":
                        result['Read Range'] = segment
                    elif segment.find("..") >= 0 and 'Read Range' not in result:
                        result['Read Range'] = segment
                    elif segment.endswith("Dynamic Range"):
                        result['Dynamic Range'] = segment.split()[0]
                    else:
                        result[segment] = True
                    last_key = None
            return result

        def _flush_read() -> None:
            """Emit the currently-open read as flat record(s), then reset state.

            Closes the read accumulated so far and appends one record per
            measured channel to ``result["reads"]``. How a read fans out into
            records depends on its optics, which arrive in one of three mutually
            exclusive shapes:

            * **Filter sets** (fluorescence/luminescence) — one record per
              ``Filter Set`` block, each parsed into Ex/Em wavelength, bandwidth
              and optics source. Fluorescence-polarization reads (signalled by
              ``Fluorescence Polarization`` in the shared metadata) get a
              ``[Parallel]``/``[Perpendicular]`` channel tag appended to the
              label by filter-set order.
            * **``Wavelengths``** (absorbance) — one record per listed
              wavelength.
            * **``Spectrum``** — a single swept record; the fixed axis is
              recorded as a wavelength, the swept axis (``Emission/Excitation
              Start``) is not.

            Reads with none of these (e.g. an absorbance endpoint with only a
            bare ``Wavelengths`` already consumed, or an unsupported read kept
            for bookkeeping) emit a single bare record. Every record is stamped
            with the read name, modality, kinetic flag, optics geometry, and the
            ``supported_*`` flags. Optics source defaults from the
            ``Optics``/``Mirror`` fields here and may be corrected later from the
            data-grid label (see ``_apply_label_optics``).

            Resets all per-read state (but not ``is_kinetic``) so the next read
            starts clean.
            """
            nonlocal read_name, read_modality, read_type, read_shared_meta, filter_sets, expecting_mode, supported_mode, supported_type
            if read_name is None:
                return
            if read_modality is None:
                log.warning(f"Read {read_name} has no modality metadata!")

            # If fluorescence/luminescence, read info is contained in "Filter Set" blocks
            reads = []
            if filter_sets:
                for i, fs in enumerate(filter_sets):
                    optics = fs.pop("Optics","")
                    mirror = fs.pop("Mirror","")
                    read_geometry = optics if optics != "" else mirror if mirror != "" else ""

                    ex:list[str] = [x.strip() for x in fs.pop("Excitation", "").split("/")]
                    em:list[str] = [x.strip() for x in fs.pop("Emission", "").split("/")]

                    ex_wv, ex_bw = '', ''
                    ex_source = ''

                    em_wv, em_bw = '', ''
                    em_source = ''

                    if mirror == "" and optics != "":
                        ex_source = FILTER
                        em_source = FILTER
                    if optics == "" and mirror != "":
                        ex_source = MONOCHROMATOR
                        em_source = MONOCHROMATOR

                    label = None
                    if read_modality == ReadModality.FLUORESCENCE:
                        ex_label, em_label = '', ''
                        if len(ex) == 0:
                            log.warning(f"Read {read_name} filter set '{fs.get('Filter Set', '')}' was a fluorescence read but there was no excitation wavelength specified.")
                        elif len(ex) == 1:
                            ex_wv = ex[0]
                            ex_label = ex_wv
                        elif len(ex) == 2:
                            ex_wv, ex_bw = ex
                            ex_label = f"{ex_wv}/{ex_bw}"
                        else:
                            log.error(f"Read {read_name} filter set '{fs.get('Filter Set', '')}': Unable to parse excitation parameters: {ex}")
                        if len(em) == 0:
                            log.warning(f"Read {read_name} filter set '{fs.get('Filter Set', '')}' was a fluorescence read but there was no emission wavelength specified.")
                        elif len(em) == 1:
                            em_wv = em[0]
                            em_label = em_wv
                        elif len(em) == 2:
                            em_wv, em_bw = em
                            em_label = f"{em_wv}/{em_bw}"
                        label = f"{ex_label},{em_label}"
                    elif read_modality == ReadModality.LUMINESCENCE:
                        label = "Lum"
                        if len(em) == 0:
                            log.warning(f"Read {read_name} filter set '{fs.get('name')}' was a luminescence read but there was no emission information specified.")
                        elif len(em) == 1:
                            if em[0] == "Hole":
                                em_source = "hole"
                            elif em[0] == "Full light":
                                em_wv = "Full light"
                            elif em[0].isnumeric():
                                em_wv = em[0]
                                label = em_wv
                        elif len(em) == 2:
                            em_wv, em_bw = em
                            label = f"{em_wv}/{em_bw}"

                    if label is not None:
                        if 'Fluorescence Polarization' in read_shared_meta:
                            if i == 0:
                                label = f"{label} [Parallel]"
                            elif i == 1:
                                label = f"{label} [Perpendicular]"
                        reads.append({
                            "label": label,
                            PlateReaderDataColumn.EXCITATION: ex_wv,
                            PlateReaderDataColumn.EXCITATION_BW: ex_bw,
                            PlateReaderDataColumn.EXCITATION_OPTICS: ex_source,
                            PlateReaderDataColumn.EMISSION: em_wv,
                            PlateReaderDataColumn.EMISSION_BW: em_bw,
                            PlateReaderDataColumn.EMISSION_OPTICS: em_source,
                            PlateReaderDataColumn.GEOMETRY: read_geometry,
                            **fs # should include read height, gain and geometry
                        })
            elif "Wavelengths" in read_shared_meta:
                wavelengths_arr = read_shared_meta.pop('Wavelengths').split(",")
                for wavelength in wavelengths_arr:
                    wv = wavelength.strip()
                    reads.append(
                        {
                            "label": wv,
                            PlateReaderDataColumn.ABSORBANCE_WV: wv,
                        })
            elif read_type == "Spectrum":
                label = None

                read_dict = {}
                if read_modality == ReadModality.ABSORBANCE:
                    label = "Spectrum"
                elif read_modality == ReadModality.FLUORESCENCE:
                    if "Emission Start" in read_shared_meta:
                        label = "EM Spectrum"
                        # Fixed excitation wavelength; emission is the swept axis.
                        ex_wv, ex_bw, ex_source = _parse_wavelength_spec(read_shared_meta.pop("Fixed Excitation", ""))
                        _, em_bw, em_source = _parse_wavelength_spec(read_shared_meta.pop("Emission Start", ""), keep_wavelength=False)
                        read_dict = {
                            PlateReaderDataColumn.EXCITATION: ex_wv,
                            PlateReaderDataColumn.EXCITATION_BW: ex_bw,
                            PlateReaderDataColumn.EXCITATION_OPTICS: ex_source,
                            PlateReaderDataColumn.EMISSION_BW: em_bw,
                            PlateReaderDataColumn.EMISSION_OPTICS: em_source,
                        }
                    elif "Excitation Start" in read_shared_meta:
                        label = "EX Spectrum"
                        # Excitation is the swept axis; emission is fixed.
                        _, ex_bw, ex_source = _parse_wavelength_spec(read_shared_meta.pop("Excitation Start", ""), keep_wavelength=False)
                        em_wv, em_bw, em_source = _parse_wavelength_spec(read_shared_meta.pop("Fixed Emission", ""))
                        read_dict = {
                            PlateReaderDataColumn.EXCITATION_BW: ex_bw,
                            PlateReaderDataColumn.EXCITATION_OPTICS: ex_source,
                            PlateReaderDataColumn.EMISSION: em_wv,
                            PlateReaderDataColumn.EMISSION_BW: em_bw,
                            PlateReaderDataColumn.EMISSION_OPTICS: em_source,
                        }
                elif read_modality == ReadModality.LUMINESCENCE:
                    if "Emission Start" in read_shared_meta:
                        label = "EM Spectrum"
                        _, em_bw, em_source = _parse_wavelength_spec(read_shared_meta.pop("Emission Start", ""), keep_wavelength=False)
                        read_dict = {
                            PlateReaderDataColumn.EMISSION_BW: em_bw,
                            PlateReaderDataColumn.EMISSION_OPTICS: em_source,
                        }

                if label is not None:
                    spectrum_dict = {
                        'label':label,
                        **read_dict
                    }
                    # Iterate read_shared_meta (insertion order) rather than the
                    # set intersection, so column order is deterministic across
                    # processes (set iteration order varies with the hash seed).
                    for k in [k for k in read_shared_meta if k in _SPECTRUM_READ_DATA]:
                        spectrum_dict[k] = read_shared_meta.pop(k)
                    reads.append(spectrum_dict)

            optics = read_shared_meta.pop("Optics","")
            mirror = read_shared_meta.pop("Mirror","")
            read_geometry = optics if optics != "" else mirror if mirror != "" else ""
            source = ''
            if mirror == "" and optics != "":
                source = FILTER
            if optics == "" and mirror != "":
                source = MONOCHROMATOR

            if len(reads) > 0:
                for r in reads:
                    read_dict = {
                        PlateReaderDataColumn.READ_NAME: read_name,
                        PlateReaderDataColumn.READ_MODALITY: read_modality,
                        # Diagnostic columns; dropped unless keep_diagnostic_metadata.
                        "read_type": read_type,
                        "kinetic": is_kinetic,
                        "supported_mode": supported_mode,
                        "supported_type": supported_type,
                        PlateReaderDataColumn.GEOMETRY: read_geometry,
                        PlateReaderDataColumn.EMISSION_OPTICS: source,
                        PlateReaderDataColumn.EXCITATION_OPTICS: source
                    }
                    result["reads"].append(read_dict | r | read_shared_meta)
            else:
                result["reads"].append({
                    PlateReaderDataColumn.READ_NAME: read_name,
                    PlateReaderDataColumn.READ_MODALITY: read_modality,
                    "read_type": read_type,
                    "supported_mode": supported_mode,
                    "supported_type": supported_type,
                    "kinetic": is_kinetic,
                    **read_shared_meta,
                })
            supported_mode = False
            supported_type = False
            read_name = None
            read_modality = None
            read_type = None
            read_shared_meta = {}
            filter_sets = []
            expecting_mode = False

        def _is_mode_token(words: list[str]) -> bool:
            """True if ``words`` begins with a recognized read modality
            (e.g. ``['Fluorescence', 'Endpoint']``), supported or not. Used to
            tell whether a line carries the read's mode/type or is just a name."""
            return bool(words) and words[0] in _SUPPORTED_READ_MODES | _NOT_SUPPORTED_READ_MODES

        def _set_read_mode(words: list[str]) -> None:
            """Set modality/type and the supported-mode/type flags from a
            mode-bearing token list (e.g. ['Fluorescence', 'Endpoint'])."""
            nonlocal read_modality, read_type, expecting_mode, supported_mode, supported_type
            read_modality = words[0]
            read_type = ' '.join(words[1:]) if len(words) > 1 else None
            expecting_mode = False
            supported_mode = read_modality in _SUPPORTED_READ_MODES
            supported_type = supported_mode and (read_type in _SUPPORTED_READ_TYPES)

        for line in lines:

            stripped = line.strip()
            # Empty line
            if not stripped:
                continue

            # Load plate type
            if stripped.startswith("Plate Type\t"):
                if result[MetadataColumn.PLATE_TYPE] is None:
                    result[MetadataColumn.PLATE_TYPE] = stripped.split("\t", 1)[1]
                    continue
                else:
                    log.warning("Duplicate 'Plate Type' entry, skipping.")

            # Parse Read information
            if stripped.startswith("Read\t"):
                _flush_read() # add previously stored Read information into list
                # Sometimes unnamed reads are referred to as "Read <#>"
                # (when there are multiple of the same reading mode + type like "Fluorescence Endpoint")
                read_number += 1
                # Get either name of read or read type
                name_part = stripped.split("\t", 1)[1].strip()
                words = name_part.split()
                if _is_mode_token(words):
                    # Mode keyword is on the Read line — no explicit name; auto-number
                    read_name = f"Read {read_number}"
                    _set_read_mode(words)
                else:
                    read_name = name_part
                    expecting_mode = True  # Need to read the next line to get the read mode
                continue

            # Read block ends if we see another action
            if stripped.startswith(_ACTIONS):
                _flush_read()
                # Track whether we are entering or exiting a kinetic section
                if stripped.startswith("Start Kinetic\t"):
                    is_kinetic = True
                elif stripped.startswith("End Kinetic"):
                    is_kinetic = False
                continue

            # If we are not within a read block, keep looking for one
            if read_name is None:
                continue

            if not expecting_mode and (not supported_mode or not supported_type):
                log.info(f"Read '{read_name}' is of unsupported mode or type '{read_modality} {read_type}'")
                # Don't parse any more?
                _flush_read()
                continue

            # Filter-set detail lines are indented with a tab followed by spaces.
            # (Heuristic — no known counterexample, but not exhaustively verified.)
            is_filter_detail = line.startswith("\t    ")

            # First content line after Read\t<name>: capture read mode and type
            if expecting_mode and not is_filter_detail:
                words = stripped.split()
                if _is_mode_token(words):
                    _set_read_mode(words)
                    continue
                else:
                    expecting_mode = False
                    # fall through — this line isn't a mode descriptor

            if stripped.startswith("Filter Set"):
                filter_sets.append({"Filter Set": stripped})
                continue

            if is_filter_detail and filter_sets:
                filter_sets[-1].update(_parse_kv_line(stripped))
                continue

            # Read-level shared metadata
            kv = _parse_kv_line(stripped)
            if kv:
                read_shared_meta.update(kv)

        _flush_read()
        return result

    # ------------------------------------------------------------------
    # Data-segment assembly
    # ------------------------------------------------------------------

    @classmethod
    def _assemble_result(
        cls,
        result: pd.DataFrame,
        metadata_read_map: dict,
        read_record_df: pd.DataFrame,
        metadata: dict,
        seg_type: str | None,
        keep_diagnostic_metadata: bool,
    ) -> PlateReaderData:
        """Assemble one parsed data segment into a ``PlateReaderData``.

        Attaches read metadata onto the data rows (via per-read-name lookup
        into ``read_record_df``), folds the per-row spectrum wavelength into
        the matching wavelength column, applies the per-axis optics ground
        truth from the data-grid label, coerces wavelength columns to numeric,
        drops diagnostic / all-empty columns, and builds the
        ``PlateReaderData`` of the appropriate format.
        """
        read_type = 'kinetic' if seg_type == 'kinetic' else 'spectrum' if seg_type == 'spectrum' else 'endpoint'
        metadata_columns = set(metadata.keys())
        if len(metadata_read_map) > 0:
            missing_reads = set(result[PlateReaderDataColumn.READ].unique()) - set(metadata_read_map)

            for m in missing_reads:
                metadata_read_map[m] = m

            # Build a per-data-read-name row lookup from read_record_df,
            # indexed by inferred_label so each data read maps to its matched
            # metadata record.  Then assign each metadata column directly onto
            # result (preserving read_record_df column order) instead of doing
            # a pandas string-key merge.
            matched_labels = result[PlateReaderDataColumn.READ].map(metadata_read_map)
            # inferred_label is unique by construction: either "<read_name>:<label>"
            # (label always non-empty) or bare read_name for group_meta rows; the colon
            # keeps the namespaces from colliding, so set_index + map is safe.
            record_index = read_record_df.set_index('inferred_label')
            meta_cols = [c for c in read_record_df.columns if c != 'inferred_label']
            for col in meta_cols:
                result[col] = matched_labels.map(record_index[col])
            metadata_columns |= set(meta_cols)

            if seg_type == "spectrum":
                # For spectrum reads the measured wavelength is in the
                # per-row 'Wavelength' column; fold it into the matching
                # wavelength column for the masked rows (only where unset,
                # preserving any value from the read metadata), then drop it.
                def _fold_wavelength(mask, column):
                    if not mask.any():
                        return
                    wv = result.loc[mask, 'Wavelength'].astype(str)
                    if column in result.columns:
                        result.loc[mask, column] = result.loc[mask, column].fillna(wv)
                    else:
                        result.loc[mask, column] = wv
                    metadata_columns.discard(column)

                # Derive the spectrum label (EM Spectrum / EX Spectrum) from
                # the matched key's suffix (the part after the last ':') rather
                # than a separate 'inferred_label' column on the result frame.
                label_suffix = matched_labels.str.split(':').str[-1]
                em_mask = (label_suffix == 'EM Spectrum') & (result[PlateReaderDataColumn.READ_MODALITY].isin((ReadModality.FLUORESCENCE, ReadModality.LUMINESCENCE)))
                _fold_wavelength(em_mask, PlateReaderDataColumn.EMISSION)
                ex_mask = (label_suffix == 'EX Spectrum') & (result[PlateReaderDataColumn.READ_MODALITY] == ReadModality.FLUORESCENCE)
                _fold_wavelength(ex_mask, PlateReaderDataColumn.EXCITATION)
                abs_mask = (result['read_type'] == 'Spectrum') & (result[PlateReaderDataColumn.READ_MODALITY] == ReadModality.ABSORBANCE)
                _fold_wavelength(abs_mask, PlateReaderDataColumn.ABSORBANCE_WV)
                if "Wavelength" in result:
                    result = result.drop(columns=["Wavelength"])

        result = result.assign(**metadata)
        result = result.mask(result == '')

        result = cls._apply_label_optics(result, seg_type)

        for wv_col in {PlateReaderDataColumn.EMISSION, PlateReaderDataColumn.EMISSION_BW, PlateReaderDataColumn.EXCITATION, PlateReaderDataColumn.ABSORBANCE_WV}:
            if wv_col in result.columns:
                result[wv_col] = pd.to_numeric(result[wv_col], errors='coerce')
        if not keep_diagnostic_metadata:
            _diagnostic_cols = ["inferred_label", "supported_mode", "supported_type", "kinetic", "label", "read_type"]
            result = result.drop(columns=_diagnostic_cols, errors='ignore')
            metadata_columns -= set(_diagnostic_cols)
        # Drop columns that are entirely empty for this block and not
        # part of the required schema — removes cross-read/modality
        # metadata that merged in as all-NaN (e.g. optics/geometry on
        # absorbance, or spectrum-only Start/Stop on a kinetic read).
        _required = REQUIRED_DATA_COLUMNS | REQUIRED_METADATA_COLUMNS | REQUIRED_READ_COLUMNS
        _empty = [c for c in result.columns if c not in _required and result[c].isna().all()]
        if _empty:
            result = result.drop(columns=_empty)
            metadata_columns -= set(_empty)
        return PlateReaderData.create_from_data(result, metadata_columns, PlateReaderDataFormat(read_type))

    @classmethod
    def _apply_label_optics(cls, result: pd.DataFrame, seg_type: str | None) -> pd.DataFrame:
        """Override the parse-time optics guess with the authoritative label-derived value.

        The data-grid ``Read`` label is the optics ground truth: a filter read
        keeps the bandwidth (``/``) in the wavelength token; a monochromator
        read drops it.  This per-axis correction is applied here for
        fluorescence/luminescence endpoint and kinetic reads.

        Spectrum reads carry no per-row wavelength token so they are skipped;
        for those the procedure-metadata ``Optics``/``Mirror`` fields (set at
        parse time in ``_flush_read``) remain the best available estimate.
        """
        if seg_type == "spectrum" or PlateReaderDataColumn.READ_MODALITY not in result.columns:
            return result
        fl_mask = result[PlateReaderDataColumn.READ_MODALITY].isin(
            (ReadModality.FLUORESCENCE, ReadModality.LUMINESCENCE)
        )
        wl_part = result[PlateReaderDataColumn.READ].astype(str).str.rpartition(":").iloc[:, 2].str.rstrip("]")
        tokens = wl_part.str.split(",")
        for axis_idx, optics_col, wv_col in (
            (0, PlateReaderDataColumn.EXCITATION_OPTICS, PlateReaderDataColumn.EXCITATION),
            (1, PlateReaderDataColumn.EMISSION_OPTICS, PlateReaderDataColumn.EMISSION),
        ):
            if optics_col not in result.columns or wv_col not in result.columns:
                continue
            tok = tokens.str[axis_idx]
            axis_mask = fl_mask & tok.notna() & result[wv_col].notna()
            if axis_mask.any():
                result.loc[axis_mask, optics_col] = tok[axis_mask].str.contains("/").map(
                    {True: FILTER, False: MONOCHROMATOR}
                )
        return result

    # ------------------------------------------------------------------
    # Read matching
    # ------------------------------------------------------------------

    @classmethod
    def _match_read(
        cls,
        read_to_match: str,
        reference_reads: Collection[str],
        metadata_read_map: dict,
    ) -> str | None:
        """Match a data read name to the procedure metadata, recording the
        match in ``metadata_read_map``. Returns the matched metadata key, or None."""
        matched_name = cls._get_closest_match(read_to_match, reference_reads)
        if matched_name is not None:
            metadata_read_map[read_to_match] = matched_name
        return matched_name

    @classmethod
    def _get_closest_match(cls, read_to_match: str, reference_reads: Collection[str]) -> str | None:
        """Find the reference read (``<Read Name>:<Wavelengths>``) that best
        matches ``read_to_match``, or None if nothing matched.

        ``reference_reads`` is the collection of reference labels to match
        against; a mapping such as ``reference_read_map`` may be passed, in which
        case only its keys (the labels) are consulted.

        Resolution order (tiers):

        1. **Exact**: candidate whose normalized ``ReadKey`` equals the data
           ``ReadKey`` (name, wavelengths, tag all equal).
        2. **Wavelength + tag** (nameless data only): when the data name is empty
           (bare wavelength form, no colon), match candidates on
           ``(wavelengths, tag)``.
        3--5. **Legacy fallback**: the existing substring-containment logic
           (``_match_substrs_ind``): name → wavelength → longest-name → None.
           Tiers 1--2 only win when they uniquely resolve a channel-tag
           ambiguity; all other cases fall through to legacy.
        """
        candidates = list(reference_reads)

        # --- Tier 1: Exact ReadKey match (name + wavelengths + tag) ----------
        # Only activate when the tag field is non-empty on EITHER the data read
        # or at least one candidate, meaning a channel-tag is in play.  This
        # keeps non-fluorescence polarization (FP) reads on the legacy path even when their ReadKeys happen
        # to match exactly (avoids any tiebreak divergence).
        data_rk = _normalize_read_key(read_to_match)
        cand_rks = [_normalize_read_key(c) for c in candidates]
        any_tag = data_rk.tag or any(rk.tag for rk in cand_rks)
        if any_tag:
            exact = [c for c, rk in zip(candidates, cand_rks) if rk == data_rk]
            if len(exact) == 1:
                log.debug(
                    "Tier 1 exact match: %r -> %r", read_to_match, exact[0]
                )
                return exact[0]

        # --- Tier 2: Wavelength + tag for nameless data reads ----------------
        # Only activate when: (a) the data read has no name (pure wavelength
        # form), and (b) channel tags are in play among the candidates.
        if data_rk.name == "" and any_tag:
            wl_tag_matches = [
                c for c, rk in zip(candidates, cand_rks)
                if rk.wavelengths == data_rk.wavelengths
                and rk.tag == data_rk.tag
            ]
            if len(wl_tag_matches) == 1:
                log.debug(
                    "Tier 2 wavelength+tag match: %r -> %r",
                    read_to_match, wl_tag_matches[0],
                )
                return wl_tag_matches[0]

        # --- Tiers 3–5: Legacy substring fallback ----------------------------
        read_names, wavelengths = (
            [''] * len(candidates),
            [''] * len(candidates),
        )
        for i, r in enumerate(candidates):
            read_names[i], _, wavelengths[i] = r.rpartition(":")

        def full_read_name(i: int):
            return f"{read_names[i]}:{wavelengths[i]}"

        if ":" not in read_to_match:
            name_split = read_to_match
            wavelength_split = read_to_match
        else:
            name_split, _, wavelength_split = read_to_match.rpartition(":")

        matches_name_ind = sorted(_match_substrs_ind(read_to_match, read_names), key=lambda i: len(read_names[i]), reverse=True)
        if len(matches_name_ind) > 0:
            if log.level == logging.DEBUG:
                log.debug(f"Read '{name_split}' matched the following read names: {set([read_names[i] for i in matches_name_ind])}")
            read_names = [read_names[i] for i in matches_name_ind]
            wavelengths = [wavelengths[i] for i in matches_name_ind]
        else:
            log.debug(f"Could not match read '{read_to_match}' (via read name '{name_split}') to any read names. Checking wavelengths.")

        read_wvs = [f"{_get_base_wl(k)}" for k in wavelengths]
        # Match wavelengths on the subset with matching names: get the indices of
        # the matched reference reads, then sort by length, longest first.
        matched_wv = sorted(_match_substrs_ind(_get_base_wl(wavelength_split), read_wvs),
                            key=lambda i: len(wavelengths[i]), reverse=True)
        if len(matched_wv) > 0:
            # then return the longest matching read name
            if len(matched_wv) > 1:
                log.warning(f"Read name '{read_to_match}' matched the following reads: {[full_read_name(i) for i in matched_wv]}. Assigning to '{full_read_name(matched_wv[0])}'.")
            log.debug(f"Matched read '{read_to_match}' to '{full_read_name(matched_wv[0])}'")
            return full_read_name(matched_wv[0])

        # we could match a name; if we can't match any wavelengths, then keep the longest matching name
        if len(matches_name_ind) > 0:
            # read_names was reordered longest-first above, so [0] is the longest
            # matching name (do NOT re-index with matches_name_ind, which holds
            # the pre-reorder positions).
            matched_name = read_names[0]
            log.warning(f"Read name '{read_to_match}' could not match any wavelengths, but found matching reads: {set(read_names)}. Assigning to the longest: {read_names[0]}")
            return matched_name
        log.warning(f"Could not find any matching reads for '{read_to_match}' in the procedural metadata.")
        return None



# ------------------------------------------------------------------
# Read identity & matching helpers
# ------------------------------------------------------------------

class ReadKey(NamedTuple):
    """Immutable, comparable representation of a Gen5 read name.

    ``name``        — human label stripped of decoration (e.g. ``"GFP-G50"``,
                      ``"Read 1"``); empty string when the raw string has no
                      colon (bare wavelength-only form).
    ``wavelengths`` — tuple of bandwidth-stripped tokens in order
                      (e.g. ``("485", "528")``).
    ``tag``         — polarization channel (``"Parallel"``, ``"Perpendicular"``)
                      or empty string for non-FP reads.
    """

    name: str
    wavelengths: tuple[str, ...]
    tag: str

# Known statistic/transform wrappers that may bracket an inner key.
# (probably not exhaustive)
_KNOWN_STAT_WRAPPERS: frozenset[str] = frozenset({
    "Corrected",
    "Max V",
    "t at Max V",
    "Mean OD",
    "R-Squared",
    "Lagtime",
    "Integral",
})

# Known polarization channel tags (trailing ``[Tag]`` on FP reads).
_KNOWN_CHANNEL_TAGS: frozenset[str] = frozenset({"Parallel", "Perpendicular"})

# Regex: trailing " [<tag>]" where <tag> is a known channel tag.
_CHANNEL_TAG_RE = re.compile(
    r"\s+\[(" + "|".join(re.escape(t) for t in _KNOWN_CHANNEL_TAGS) + r")\]\s*$"
)

# Regex: "<prefix> [<inner>]" wrapper form (prefix has no colon).
_BRACKET_WRAPPER_RE = re.compile(r"^([^\[:\]]+)\[([^\]]+)\]\s*$")

def _normalize_read_key(s: str) -> ReadKey:
    """Normalize a raw Gen5 read-name string into a ``ReadKey``.

    Works uniformly on **both** sides of the matching problem:

    * Data-table read names (e.g. ``"Blank 630nmAbsRead:630"``,
      ``"485,528"``, ``"Max V [Read 2:450,490]"``)
    * Procedure-metadata keys (e.g. ``"Read 1:485/20,528/20"``,
      ``"FP Read:485/20,528/20 [Parallel]"``)

    Normalization steps (in order):

    1. Strip a trailing known channel tag → ``tag``.
    2. Detect a ``<stat> [<inner>]`` wrapper (stat is a known name with no
       colon). If recognized, recurse on ``<inner>``.
    3. Strip a leading ``Blank `` prefix.
    4. Split on the **last** colon (``rpartition(":")``):
       - If a colon is present: left → ``name``, right → wavelength spec.
       - If no colon: empty ``name``, whole string → wavelength spec.
    5. Parse the wavelength spec into individual bandwidth-stripped tokens.
    """
    s = s.strip()

    # --- Step 1: strip trailing known channel tag -------------------------
    tag = ""
    m = _CHANNEL_TAG_RE.search(s)
    if m:
        tag = m.group(1)
        s = s[: m.start()].rstrip()

    # --- Step 2: detect bracket wrapper -----------------------------------
    bm = _BRACKET_WRAPPER_RE.match(s)
    if bm:
        prefix = bm.group(1).rstrip()
        inner = bm.group(2).strip()
        if prefix in _KNOWN_STAT_WRAPPERS:
            # Recognized stat wrapper: recurse on inner, preserve outer tag.
            inner_key = _normalize_read_key(inner)
            return ReadKey(
                name=inner_key.name,
                wavelengths=inner_key.wavelengths,
                tag=tag or inner_key.tag,
            )
        else:
            # Unrecognized wrapper: do NOT unwrap; emit a visible warning.
            log.warning(
                "Unrecognized bracketed wrapper %r in read-key %r — "
                "name will not be unpeeled. If this is a known statistic or "
                "transform, add it to _KNOWN_STAT_WRAPPERS.",
                prefix,
                s,
            )
            # Fall through: parse the original string as-is.

    # --- Step 3: strip leading 'Blank ' prefix ----------------------------
    if s.startswith("Blank "):
        s = s[len("Blank "):]

    # --- Step 4: split on last colon --------------------------------------
    # rpartition: when no ":" found → head="", sep="", tail=original
    head, sep, tail = s.rpartition(":")
    if sep:
        name = head.strip()
        wl_part = tail.strip()
    else:
        # No colon — entire string is the wavelength part.
        name = ""
        wl_part = tail.strip()  # tail holds the original string when sep==""

    # Strip a trailing ']' left over from unrecognized wrappers.
    wl_part = wl_part.rstrip("]").rstrip()

    # --- Step 5: parse wavelength tokens ----------------------------------
    if "," in wl_part:
        wavelengths = tuple(
            _get_base_wl(tok.strip())
            for tok in wl_part.split(",")
            if tok.strip()
        )
    elif wl_part:
        wavelengths = (_get_base_wl(wl_part),)
    else:
        wavelengths = ()

    return ReadKey(name=name, wavelengths=wavelengths, tag=tag)

def _get_base_wl(wl_str: str) -> str:
    """Strip filter bandwidth, e.g. '485/20' -> '485'."""
    return ",".join(w.split("/")[0].strip() for w in wl_str.split(","))

def _match_substrs_ind(superstring: str, substrings: Collection[str]) -> list[int]:
    """Indices of the ``substrings`` that occur within ``superstring``.

    The containment primitive behind the legacy (tier 3-5) read-name matcher:
    e.g. a data read ``"GFP-G70:485,528"`` contains the procedure read name
    ``"GFP-G70"``, so that candidate's index is returned.
    """
    return [ind for ind, sub in enumerate(substrings) if sub in superstring]

def _parse_wavelength_spec(text: str, keep_wavelength: bool = True) -> tuple[str, str, str]:
    """Parse a Gen5 wavelength spec into ``(wavelength, bandwidth, optics)``.

    One token (e.g. ``"485"``) → monochromator with no bandwidth; two tokens
    (``"485/20"``) → filter with that bandwidth. A trailing `` nm`` is stripped.
    ``keep_wavelength=False`` is for swept axes (Emission/Excitation *Start*),
    where the single value is the sweep start rather than a fixed wavelength and
    so is not recorded.
    """
    parts = [p.strip() for p in text.removesuffix(" nm").split("/")]
    if len(parts) == 2:
        return (parts[0] if keep_wavelength else ''), parts[1], FILTER
    if len(parts) == 1:
        return (parts[0] if keep_wavelength else ''), '', MONOCHROMATOR
    return '', '', ''


# ------------------------------------------------------------------
# Segment & table parsing
# ------------------------------------------------------------------

def _split_segments(lines: list[str]) -> list[list[str]]:
    """Split lines into non-empty groups separated by blank lines."""
    segments: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip() == "":
            if current:
                segments.append(current)
                current = []
        else:
            current.append(line)
    if current:
        segments.append(current)
    return segments

_WELL_ID_RE = re.compile(r"[A-Z]{1,2}\d{1,2}")

def _looks_like_rowwise_results_table(seg: list[str]) -> bool:
    """Detect a Gen5 tabular ("statistics") results export.

    These list one row per well — ``<Sample>\\t<Well>\\t<Conc/Dil>\\t<value>...`` —
    rather than the supported plate-grid ("Matrix") layout, whose leading line is
    a tab-led header of plate column numbers. The tell is that the segment's first
    line is tab-delimited data whose second field is a well ID (e.g. 'A4', 'H12'),
    which never occurs in a grid's header line.
    """
    if not seg:
        return False
    fields = seg[0].split("\t")
    return len(fields) >= 2 and _WELL_ID_RE.fullmatch(fields[1].strip()) is not None

def parse_data_grid(seg: list[str],
                    read_column_name: str = PlateReaderDataColumn.READ,
                    data_column_name: str = PlateReaderDataColumn.DATA) -> pd.DataFrame | None:
    """Parse a Gen5 plate-grid segment into a long-form Well/Data/Read frame.

    A grid is a plate matrix: the header line is ``<TAB>1<TAB>2…`` (plate column
    numbers), and each subsequent line is ``<Row><TAB>v1<TAB>v2…<TAB><Read>``.
    Multiple reads of the same plate share rows — the row label appears once and
    is blank on continuation lines (hence the ``ffill`` below), and the read
    identifier sits in the *trailing* column. The grid is melted from
    wide (one column per plate column) to long (one row per well per read),
    yielding ``[Well, <data>, <read>]``.

    Returns ``None`` if the segment is empty or unparseable.
    """
    if len(seg) < 1:
        return None

    col_headers = seg[0].split("\t")
    if len(col_headers) == 0:
        return None
    col_headers[0] = 'Row'
    # The read identifier is the unlabeled trailing column; name it explicitly
    # so melt keeps it as an id_var rather than treating it as a plate column.
    col_headers = col_headers + [read_column_name,]
    try:
        df = pd.read_table(io.StringIO("\n".join(seg)),sep='\t', names=col_headers,header=0,index_col=False)
        # Row label is written once per plate-row and blank on the continuation
        # lines for that row's other reads; carry it down so every read row
        # knows its plate row.
        df['Row'] = df['Row'].ffill(axis=0)
        df_melted = df.melt(id_vars = ["Row",read_column_name], var_name='Column',value_name = data_column_name)
        if not pd.api.types.is_numeric_dtype(df_melted[data_column_name]):
            log.warning("Data column contains non-numeric values.")
        df_melted['Well'] = df_melted['Row'] + df_melted['Column']

        # Drop values that were not recorded
        df_melted = df_melted.dropna(subset=[data_column_name])
        # Coerce to numeric; invalid sentinels (_INVALID) and other
        # non-numeric values (e.g. "<", ">") become NaN.
        df_melted[data_column_name] = pd.to_numeric(
            df_melted[data_column_name].astype(str).str.replace(
                ",", "", regex=False
            ),
            errors='coerce',
        )
        df_melted[read_column_name] = df_melted[read_column_name].astype("str")
        return df_melted[['Well',data_column_name, read_column_name]]
    except Exception as e:
        # Broad by design: a malformed grid should be skipped, not abort the
        # whole file load. The raisers are heterogeneous — pandas
        # ParserError/EmptyDataError (both ValueError) from read_table, plus
        # AttributeError/KeyError from melt on adversarial headers (e.g. a
        # column literally named "Read"/"Row") — so enumerating them buys
        # little. Golden characterization tests guard against masked bugs.
        log.warning(f"Failed to parse the data grid: {e}")
        return None

def parse_data_table(
        seg: list[str],
        data_column_name: str = PlateReaderDataColumn.DATA,
        temperature_column_name: str = PlateReaderDataColumn.TEMPERATURE) -> pd.DataFrame | None:
    """Parse a tab-separated read table (leading Time or Well column, optional
    Temperature column) into a long-form Well/Data frame."""
    try:
        df = pd.read_table(io.StringIO("\n".join(seg))).dropna(axis=1, how="all")
    except Exception as e:
        # Broad by design, as in parse_data_grid: skip an unparseable table
        # rather than abort the whole file load.
        log.warning(f"Pandas failed to parse the data table: {e}")
        return None
    if len(df.columns) == 0:
        log.warning("Parsed data table was empty.")
    # Should be "Time" or "Well"
    if df.columns[0] == "Well":
        df = df.rename(columns={"Well":PlateReaderDataColumn.READ})
    elif df.columns[0] == "Time":
        df = df.rename(columns={"Time":PlateReaderDataColumn.TIME})
        # Gen5 writes kinetic times as MM:SS for short runs and H:MM:SS once
        # they pass an hour. pd.to_timedelta needs H:MM:SS, so promote the
        # one-colon (MM:SS) form by prefixing "0:" — but only when *every* value
        # is MM:SS, to avoid mangling a table that already uses H:MM:SS.
        time_series = df[PlateReaderDataColumn.TIME].astype(str)
        if time_series.str.count(":").eq(1).all():
            time_series = "0:" + time_series
        df[PlateReaderDataColumn.TIME] = pd.to_timedelta(time_series, errors="coerce")
    id_vars = [df.columns[0],]
    # A second leading column starting with "T" is the per-reading temperature
    # (e.g. "T° 600"); treat it as an id_var so it isn't melted into the data.
    if len(df.columns) > 1 and df.columns[1].startswith("T"):
        df = df.rename(columns={df.columns[1]: temperature_column_name})
        id_vars.append(temperature_column_name)
    return_df = df.melt(id_vars = id_vars,var_name=PlateReaderDataColumn.WELL,value_name=data_column_name)
    return_df[data_column_name] = pd.to_numeric(
        return_df[data_column_name].astype(str).str.replace(
            ",", "", regex=False
        ),
        errors='coerce',
    )
    if PlateReaderDataColumn.READ in return_df.columns:
        return_df[PlateReaderDataColumn.READ] = return_df[PlateReaderDataColumn.READ].astype(str)
    if return_df.columns[0] == PlateReaderDataColumn.READ:
        # Move the leading Read column to the end
        cols = list(return_df.columns)
        return return_df[cols[1:] + cols[:1]]
    else:
        return return_df


# ------------------------------------------------------------------
# xlsx -> txt rendering
# ------------------------------------------------------------------

def _format_xlsx_cell(cell) -> str:
    """Render an openpyxl cell as the text Gen5 would have written to a .txt
    export. Number-format-aware so the read code `440535` -> `440,535` matches
    the procedure metadata: an integer cell whose number format contains a
    thousands separator is rendered with grouping. Dates render as `%m/%d/%Y`,
    clock times as `%I:%M:%S %p`, and 1900-epoch durations (kinetic elapsed
    time) as `H:MM:SS`."""
    v = cell.value
    if v is None:
        return ""
    fmt = cell.number_format or ""
    if isinstance(v, dt.datetime):
        if v.year != 1900:
            return v.strftime("%m/%d/%Y")
        if "AM/PM" in fmt.upper():
            return v.strftime("%I:%M:%S %p")
        delta = v - dt.datetime(1900, 1, 1)
        total = int(delta.total_seconds())
        hours, rem = divmod(total, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    if isinstance(v, dt.time):
        return v.strftime("%I:%M:%S %p")
    if isinstance(v, int) or (isinstance(v, float) and float(v).is_integer()):
        n = int(v)
        return f"{n:,}" if "," in fmt else str(n)
    return str(v)

def _trim(cells: list[str]) -> list[str]:
    """Drop trailing empty cells from a rendered row."""
    end = len(cells)
    while end > 0 and cells[end - 1] == "":
        end -= 1
    return list(cells[:end])

def _is_data_row(row: list[str]) -> bool:
    """A data row: column A empty AND at least 3 cells (xlsx quirk 1)."""
    return len(row) >= 3 and row[0] == ""

def _xlsx_rows_to_lines(rows) -> list[str]:
    """Convert one worksheet's openpyxl rows into the exact lines Gen5 would
    have written to a .txt export.

    Walks the rows once, accumulating contiguous runs of data rows (column A
    empty, >=3 cells) into blocks and emitting everything else verbatim. Each
    block is corrected for the ``.xlsx`` layout quirks: one leading column is
    stripped, padding rows are dropped, a grid block (header first cell empty)
    has its most-recent ``Results``/``Layout`` label re-attached, and blank
    lines are inserted around blocks (and after ``Procedure Details``) so the
    line parser sees the same segment boundaries it would in a ``.txt`` file.
    """
    rendered = [_trim([_format_xlsx_cell(c) for c in row]) for row in rows]
    lines: list[str] = []
    last_label = "Results"
    i, n = 0, len(rendered)
    while i < n:
        if _is_data_row(rendered[i]):
            block = []
            while i < n and _is_data_row(rendered[i]):
                block.append(rendered[i])
                i += 1
            block = [r[1:] for r in block]  # strip one leading column
            is_grid = block[0][0] == ""  # grid header "\t1\t2..."
            lines.append("")  # blank line before
            if is_grid:
                lines.append(last_label)  # re-attach Results/Layout
            for r in block:
                if len([c for c in r if c != ""]) < 3:
                    continue  # drop padding rows
                lines.append("\t".join(r))
            lines.append("")  # blank line after
        else:
            row = rendered[i]
            lines.append("\t".join(row))
            nonempty = [c for c in row if c != ""]
            if len(nonempty) == 1:
                last_label = nonempty[0]
            if row and row[0] == "Procedure Details":
                lines.append("")  # quirk 2
            i += 1
    return lines
