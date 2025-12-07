from pathlib import Path
from typing import List, Tuple
import re
import zipfile
from ..chapter.detector import SmartChapterDetector, AdvancedChapterProcessor
from ..performance import memory_efficient


class OptimizedFileProcessor:
    """优化的文件处理器，集成智能章节分割"""
    
    def __init__(self, config):
        self.config = config
        self.detector = None
        self.processor = None
        self._init_chapter_system()
    
    def _init_chapter_system(self):
        """初始化章节系统"""
        split_config = self.config.get("split")
        
        # 创建自定义规则
        custom_rules = []
        custom_rules_str = split_config.get("custom_rules", "")
        if custom_rules_str:
            try:
                import json
                rules_data = json.loads(custom_rules_str)
                from ..chapter.detector import ChapterRule
                for rule_data in rules_data:
                    rule = ChapterRule(**rule_data)
                    custom_rules.append(rule)
            except json.JSONDecodeError:
                from ..core.logger import logger
                logger.warning(f"自定义规则格式错误: {custom_rules_str}")
        
        # 创建检测器
        self.detector = SmartChapterDetector(custom_rules)
        
        # 创建处理器
        self.processor = AdvancedChapterProcessor(self.detector)
        self.processor.min_chapter_length = split_config.get("min_chapter_length", 100)
        self.processor.merge_short_chapters = split_config.get("merge_short_chapters", True)
    
    @memory_efficient
    def split_novel_advanced(self, src_path: Path, dst_dir: Path) -> List[Path]:
        """高级小说分割"""
        if not dst_dir.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
        
        encoding = self._detect_encoding(src_path)
        content = self._read_file_safely(src_path, encoding)
        
        if not content:
            from ..core.logger import logger
            logger.error(f"无法读取文件: {src_path}")
            return []
        
        # 使用智能章节分割
        if self.config.get("split").get("enable_smart_detection", True):
            chapters = self.processor.split_content_advanced(content, encoding)
        else:
            # 回退到传统正则分割
            chapters = self._split_by_traditional_pattern(content)
        
        if not chapters:
            from ..core.logger import logger
            logger.warning(f"未检测到章节: {src_path}")
            return []
        
        # 生成章节文件
        created_files = []
        for i, (title, chapter_content) in enumerate(chapters, 1):
            safe_title = self._sanitize_filename(title)
            if not safe_title:
                safe_title = f"第{i:03d}章"
            
            chapter_file = dst_dir / f"{i:03d}_{safe_title}.txt"
            
            try:
                with chapter_file.open("w", encoding="utf-8") as f:
                    f.write(f"{title}\n")
                    f.write("=" * len(title) * 2)
                    f.write(f"\n\n{chapter_content}\n")
                
                created_files.append(chapter_file)
                from ..core.logger import logger
                logger.info(f"创建章节文件: {chapter_file.name} ({len(chapter_content)}字符)")
                
            except Exception as e:
                from ..core.logger import logger
                logger.error(f"写入章节文件失败 {chapter_file}: {e}")
                continue
        
        from ..core.logger import logger
        logger.info(f"分割完成: {src_path.name} -> {len(created_files)} 个章节")
        return created_files
    
    def _split_by_traditional_pattern(self, content: str) -> List[Tuple[str, str]]:
        """传统正则分割"""
        pattern = re.compile(self.config.get("split").get("chapter_pattern"), re.UNICODE)
        lines = content.splitlines(keepends=True)
        
        chapters = []
        current_chapter = []
        current_title = ""
        
        for line in lines:
            match = pattern.match(line.strip())
            if match:
                # 保存前一个章节
                if current_chapter:
                    chapters.append((current_title, "".join(current_chapter).strip()))
                
                # 开始新章节
                current_title = line.strip()
                current_chapter = [line]
            else:
                current_chapter.append(line)
        
        # 保存最后一个章节
        if current_chapter:
            chapters.append((current_title, "".join(current_chapter).strip()))
        
        return chapters
    
    def _detect_encoding(self, path: Path) -> str:
        """检测文件编码"""
        for enc in self.config.get("split").get("encoding_fallback", ["utf-8"]):
            try:
                with path.open("r", encoding=enc) as f:
                    f.read(1024)
                return enc
            except (UnicodeDecodeError, UnicodeError):
                continue
        return "utf-8"
    
    def _read_file_safely(self, path: Path, encoding: str, max_size_mb: int = 100) -> str:
        """安全读取文件"""
        try:
            file_size = path.stat().st_size
            if file_size > max_size_mb * 1024 * 1024:
                from ..core.logger import logger
                logger.warning(f"文件过大 ({file_size/1024/1024:.1f}MB)，使用分块读取: {path}")
                return self._read_file_chunked(path, encoding)
            
            with path.open("r", encoding=encoding) as f:
                return f.read()
        except Exception as e:
            from ..core.logger import logger
            logger.error(f"读取文件失败 {path}: {e}")
            return ""
    
    def _read_file_chunked(self, path: Path, encoding: str, chunk_size: int = 1024*1024) -> str:
        """分块读取大文件"""
        content_parts = []
        try:
            with path.open("r", encoding=encoding) as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    content_parts.append(chunk)
                    from ..performance import memory_monitor
                    memory_monitor.check_memory()
            
            return "".join(content_parts)
        except Exception as e:
            from ..core.logger import logger
            logger.error(f"分块读取文件失败 {path}: {e}")
            return ""
    
    def _sanitize_filename(self, filename: str) -> str:
        """安全化文件名"""
        filename = re.sub(r'[<>:\"/\\|?*]', '_', filename)
        filename = filename.strip('. ')
        return filename[:100]  # 限制长度
    
    @staticmethod
    def batch_archive(src_dir: Path, dst_dir: Path, max_size_mb: int = 95) -> List[Path]:
        """批量归档"""
        if not src_dir.is_dir():
            from ..core.logger import logger
            logger.error(f"源目录不存在: {src_dir}")
            return []
        
        dst_dir.mkdir(parents=True, exist_ok=True)
        txt_files = sorted(src_dir.glob("*.txt"))
        
        if not txt_files:
            from ..core.logger import logger
            logger.warning(f"未找到txt文件: {src_dir}")
            return []
        
        archives = []
        current_archive = []
        current_size = 0
        archive_index = 1
        
        max_size_bytes = max_size_mb * 1024 * 1024
        
        for txt_file in txt_files:
            try:
                file_size = txt_file.stat().st_size
                
                # 检查是否需要新建归档
                if current_archive and (current_size + file_size > max_size_bytes):
                    archive_path = dst_dir / f"batch_{archive_index:03d}.zip"
                    OptimizedFileProcessor._create_zip_archive(current_archive, archive_path)
                    archives.append(archive_path)
                    
                    current_archive = []
                    current_size = 0
                    archive_index += 1
                
                current_archive.append(txt_file)
                current_size += file_size
                
            except Exception as e:
                from ..core.logger import logger
                logger.error(f"处理文件失败 {txt_file}: {e}")
                continue
        
        # 处理剩余的归档
        if current_archive:
            archive_path = dst_dir / f"batch_{archive_index:03d}.zip"
            OptimizedFileProcessor._create_zip_archive(current_archive, archive_path)
            archives.append(archive_path)
        
        from ..core.logger import logger
        logger.info(f"批量归档完成: {len(txt_files)} 个文件 -> {len(archives)} 个归档")
        return archives
    
    @staticmethod
    def _create_zip_archive(files: List[Path], archive_path: Path):
        """创建ZIP归档"""
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in files:
                    zf.write(file_path, file_path.name)
            from ..core.logger import logger
            logger.info(f"创建归档: {archive_path.name} ({len(files)} 个文件)")
        except Exception as e:
            from ..core.logger import logger
            logger.error(f"创建归档失败 {archive_path}: {e}")
    
    async def process_file(self, input_path: Path, output_dir: Path):
        """处理文件"""
        # 这是一个占位方法，实际处理逻辑在split_novel_advanced中
        # 这里简单调用split_novel_advanced
        self.split_novel_advanced(input_path, output_dir)
