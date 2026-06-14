import logging
from typing import Any, Dict

from src_m.pipeline.registry import PipelineStepExecutor, StepRegistry

logger = logging.getLogger(__name__)


class FanqieDownloadStep(PipelineStepExecutor):

    def get_name(self) -> str:
        return "fanqie_download"

    def get_input_type(self) -> str:
        return "any"

    def get_output_type(self) -> str:
        return "text_directory"

    async def execute(self, params: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        book_id = params.get("book_id")
        if not book_id:
            raise ValueError("fanqie_download requires 'book_id' in params")

        output_dir = params.get("output_dir", f"downloads/{book_id}")

        from src_m.extensions.fanqie.downloader_parser import update_book

        success, message = update_book(book_id, data_dir=output_dir)
        if not success:
            raise RuntimeError(f"fanqie_download failed for book_id={book_id}: {message}")

        logger.info("fanqie_download: downloaded book_id=%s to output_dir=%s", book_id, output_dir)

        return {"output_dir": output_dir}


class TextSplitStep(PipelineStepExecutor):

    def get_name(self) -> str:
        return "text_split"

    def get_input_type(self) -> str:
        return "text_file"

    def get_output_type(self) -> str:
        return "text_directory"

    async def execute(self, params: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_file = params.get("input_file") or inputs.get("output_dir")
        if not input_file:
            raise ValueError(
                "text_split requires 'input_file' in params or upstream 'output_dir' in inputs"
            )

        output_dir = "split_output/"

        logger.info(
            "text_split: would split input_file=%s to output_dir=%s",
            input_file,
            output_dir,
        )

        return {"output_dir": output_dir}


class TTSConvertStep(PipelineStepExecutor):

    def get_name(self) -> str:
        return "tts_convert"

    def get_input_type(self) -> str:
        return "text_directory"

    def get_output_type(self) -> str:
        return "audio_directory"

    async def execute(self, params: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_dir = inputs.get("output_dir")
        if not input_dir:
            raise ValueError("tts_convert requires upstream 'output_dir' in inputs")

        voice = params.get("voice", "zh-CN-XiaoxiaoNeural")
        concurrency = params.get("concurrency", 8)

        output_dir = "tts_output/"

        logger.info(
            "tts_convert: would convert input_dir=%s voice=%s concurrency=%s to output_dir=%s",
            input_dir,
            voice,
            concurrency,
            output_dir,
        )

        return {"output_dir": output_dir}


class AudioMergeStep(PipelineStepExecutor):

    def get_name(self) -> str:
        return "audio_merge"

    def get_input_type(self) -> str:
        return "audio_directory"

    def get_output_type(self) -> str:
        return "audio_file"

    async def execute(self, params: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_dir = inputs.get("output_dir")
        if not input_dir:
            raise ValueError("audio_merge requires upstream 'output_dir' in inputs")

        output_format = params.get("output_format", "mp3")
        output_file = f"merged_output.{output_format}"

        logger.info(
            "audio_merge: would merge input_dir=%s into output_file=%s",
            input_dir,
            output_file,
        )

        return {"output_file": output_file}


class AudioPostProcessStep(PipelineStepExecutor):

    def get_name(self) -> str:
        return "audio_post_process"

    def get_input_type(self) -> str:
        return "audio_file"

    def get_output_type(self) -> str:
        return "audio_file"

    async def execute(self, params: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_file = inputs.get("output_file")
        if not input_file:
            raise ValueError("audio_post_process requires upstream 'output_file' in inputs")

        effects = params.get("effects")
        if not effects:
            raise ValueError("audio_post_process requires 'effects' in params")

        output_file = "processed_output.mp3"

        logger.info(
            "audio_post_process: would apply effects=%s to input_file=%s -> output_file=%s",
            effects,
            input_file,
            output_file,
        )

        return {"output_file": output_file}


class EpubExportStep(PipelineStepExecutor):

    def get_name(self) -> str:
        return "epub_export"

    def get_input_type(self) -> str:
        return "text_directory"

    def get_output_type(self) -> str:
        return "epub_file"

    async def execute(self, params: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_dir = inputs.get("output_dir")
        if not input_dir:
            raise ValueError("epub_export requires upstream 'output_dir' in inputs")

        title = params.get("title")
        if not title:
            raise ValueError("epub_export requires 'title' in params")

        author = params.get("author")
        if not author:
            raise ValueError("epub_export requires 'author' in params")

        output_file = "output.epub"

        logger.info(
            "epub_export: would export input_dir=%s title=%s author=%s -> output_file=%s",
            input_dir,
            title,
            author,
            output_file,
        )

        return {"output_file": output_file}


def register_builtin_steps(registry: StepRegistry) -> None:
    for step_cls in (
        FanqieDownloadStep,
        TextSplitStep,
        TTSConvertStep,
        AudioMergeStep,
        AudioPostProcessStep,
        EpubExportStep,
    ):
        registry.register(step_cls())
