from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import re


class ChapterPattern(Enum):
    """章节模式枚举"""
    TRADITIONAL = "traditional"  # 传统：第X章
    NUMBERED = "numbered"        # 数字：1. 第一章
    ROMAN = "roman"             # 罗马数字：I. 第一章
    CUSTOM = "custom"           # 自定义正则

class ChapterPreset(Enum):
    """章节预设枚举"""
    DEFAULT = "default"          # 默认规则
    CHINESE_NOVEL = "chinese"    # 中文小说
    ENGLISH_NOVEL = "english"    # 英文小说

@dataclass
class ChapterRule:
    """章节分割规则"""
    name: str
    pattern: str
    description: str
    priority: int = 0
    enabled: bool = True
    
    # 智能检测选项
    check_indentation: bool = True      # 检查缩进
    check_empty_line: bool = True       # 检查空行分隔
    check_title_length: bool = True     # 检查标题长度
    max_title_length: int = 50          # 最大标题长度
    min_title_length: int = 2           # 最小标题长度
    
    # 上下文验证
    require_capital: bool = True        # 需要大写开头
    require_no_indent: bool = True      # 需要顶格
    allow_space_prefix: bool = False    # 允许空格前缀

class SmartChapterDetector:
    """智能章节检测器"""
    
    # 内置规则库
    DEFAULT_RULES = [
        ChapterRule(
            name="传统章节",
            pattern=r'^(引子|序章|前言|后记|附录|第[一二两三四五六七八九十百千万亿\d零]+[章篇章节回集部卷]\s*.*)$',
            description="匹配传统章节格式：第X章、引子、序章等",
            priority=10,
            check_indentation=True,
            require_no_indent=True
        ),
        ChapterRule(
            name="数字章节",
            pattern=r'^\s*(\d+)[\.\s]+[第]?[一二两三四五六七八九十百千万\d零]+[章篇章节回集部卷]?\s*.*$',
            description="匹配数字章节格式：1. 第一章、2. 第二章等",
            priority=8,
            check_indentation=True,
            allow_space_prefix=True
        ),
        ChapterRule(
            name="罗马数字",
            pattern=r'^\s*[IVXLCDM]+[\.\s]+.*$',
            description="匹配罗马数字章节：I. 第一章、II. 第二章等",
            priority=6,
            check_indentation=True,
            allow_space_prefix=True
        ),
        ChapterRule(
            name="卷册章节",
            pattern=r'^\s*(卷|册|部|篇|集)[\s一二两三四五六七八九十百千万\d零]+[\s：:].*$',
            description="匹配卷册格式：卷一、册二、部三等",
            priority=7,
            check_indentation=True,
            allow_space_prefix=True
        ),
        ChapterRule(
            name="特殊章节",
            pattern=r'^\s*(正文|终章|尾声|后记|附录|注释|参考文献)\s*$',
            description="匹配特殊章节：正文、终章、尾声等",
            priority=5,
            check_indentation=True,
            allow_space_prefix=True
        )
    ]
    
    def __init__(self, custom_rules: List[ChapterRule] = None):
        self.rules = self.DEFAULT_RULES.copy()
        if custom_rules:
            self.rules.extend(custom_rules)
        
        # 按优先级排序
        self.rules.sort(key=lambda x: x.priority, reverse=True)
        
        # 编译正则表达式
        self._compiled_patterns = {}
        for rule in self.rules:
            if rule.enabled:
                try:
                    self._compiled_patterns[rule.name] = re.compile(rule.pattern, re.UNICODE)
                except re.error as e:
                    from ..core.logger import logger
                    logger.error(f"规则 {rule.name} 的正则表达式无效: {e}")
    
    def analyze_context(self, lines: List[str], current_idx: int) -> Dict[str, Any]:
        """分析上下文信息"""
        context = {
            'prev_line': lines[current_idx - 1] if current_idx > 0 else '',
            'current_line': lines[current_idx],
            'next_line': lines[current_idx + 1] if current_idx < len(lines) - 1 else '',
            'prev_empty': False,
            'next_empty': False,
            'current_indent': 0,
            'next_indent': 0
        }
        
        # 检查空行
        context['prev_empty'] = not context['prev_line'].strip()
        context['next_empty'] = not context['next_line'].strip()
        
        # 计算缩进
        if context['current_line']:
            context['current_indent'] = len(context['current_line']) - len(context['current_line'].lstrip())
        if context['next_line']:
            context['next_indent'] = len(context['next_line']) - len(context['next_line'].lstrip())
        
        return context
    
    def validate_chapter_title(self, title: str, context: Dict[str, Any], rule: ChapterRule) -> bool:
        """验证章节标题"""
        # 长度检查
        if rule.check_title_length:
            title_len = len(title.strip())
            if title_len < rule.min_title_length or title_len > rule.max_title_length:
                return False
        
        # 大写开头检查
        if rule.require_capital and title.strip():
            first_char = title.strip()[0]
            if not (first_char.isupper() or ord(first_char) > 127):  # 中文或其他语言
                return False
        
        # 顶格检查
        if rule.require_no_indent and context['current_indent'] > 0:
            return False
        
        # 空格前缀检查
        if not rule.allow_space_prefix and title.startswith(' '):
            return False
        
        # 空行分隔检查
        if rule.check_empty_line and not (context['prev_empty'] or context['next_empty']):
            # 如果没有空行分隔，检查是否是明显的章节标题
            if not any(keyword in title for keyword in ['第', '章', '节', '篇', '卷']):
                return False
        
        return True
    
    def detect_chapter(self, lines: List[str], start_idx: int = 0) -> Optional[Tuple[int, str, ChapterRule]]:
        """智能检测章节"""
        for i in range(start_idx, len(lines)):
            line = lines[i]
            if not line.strip():
                continue
            
            context = self.analyze_context(lines, i)
            
            # 尝试每个规则
            for rule in self.rules:
                if not rule.enabled or rule.name not in self._compiled_patterns:
                    continue
                
                pattern = self._compiled_patterns[rule.name]
                match = pattern.match(line.strip())
                
                if match and self.validate_chapter_title(line.strip(), context, rule):
                    return (i, line.strip(), rule)
        
        return None
    
    def get_chapter_boundaries(self, lines: List[str]) -> List[Tuple[int, str, ChapterRule]]:
        """获取所有章节边界"""
        boundaries = []
        current_idx = 0
        
        while current_idx < len(lines):
            result = self.detect_chapter(lines, current_idx)
            if result:
                boundaries.append(result)
                current_idx = result[0] + 1
            else:
                break
        
        return boundaries

class AdvancedChapterProcessor:
    """高级章节处理器"""
    
    def __init__(self, detector: SmartChapterDetector = None):
        self.detector = detector or SmartChapterDetector()
        self.min_chapter_length = 100  # 最小章节长度（字符）
        self.merge_short_chapters = True  # 合并短章节
    
    def split_content_advanced(self, content: str, encoding: str = 'utf-8') -> List[Tuple[str, str]]:
        """高级内容分割"""
        lines = content.splitlines(keepends=True)
        boundaries = self.detector.get_chapter_boundaries(lines)
        
        if not boundaries:
            # 如果没有检测到章节，按固定长度分割
            return self._split_by_length(content)
        
        chapters = []
        total_lines = len(lines)
        
        for i, (line_idx, title, rule) in enumerate(boundaries):
            # 计算章节内容范围
            start_line = line_idx
            if i + 1 < len(boundaries):
                end_line = boundaries[i + 1][0]
            else:
                end_line = total_lines
            
            # 提取章节内容
            chapter_lines = lines[start_line:end_line]
            chapter_content = ''.join(chapter_lines).strip()
            
            # 清理章节标题
            clean_title = self._clean_chapter_title(title, rule)
            
            # 验证章节长度
            if len(chapter_content) < self.min_chapter_length:
                if self.merge_short_chapters and chapters:
                    # 合并到前一个章节
                    prev_title, prev_content = chapters[-1]
                    chapters[-1] = (prev_title, prev_content + '\n\n' + chapter_content)
                else:
                    chapters.append((clean_title, chapter_content))
            else:
                chapters.append((clean_title, chapter_content))
        
        return chapters
    
    def _clean_chapter_title(self, title: str, rule: ChapterRule) -> str:
        """清理章节标题"""
        # 移除多余的空格和标点
        title = title.strip()
        
        # 移除行尾标点
        title = re.sub(r'[：:;；,.，。！？\s]+$', '', title)
        
        # 标准化格式
        if '第' in title and '章' in title:
            # 标准化传统章节格式
            title = re.sub(r'\s+', ' ', title)
        
        return title
    
    def _split_by_length(self, content: str, max_length: int = 50000) -> List[Tuple[str, str]]:
        """按长度分割内容"""
        if len(content) <= max_length:
            return [("全文", content)]
        
        parts = []
        start = 0
        part_num = 1
        
        while start < len(content):
            end = min(start + max_length, len(content))
            
            # 尝试在句子边界分割
            if end < len(content):
                # 查找最近的句子结束
                for i in range(end, start, -1):
                    if content[i] in '。！？.!?':
                        end = i + 1
                        break
            
            part_content = content[start:end].strip()
            if part_content:
                parts.append((f"第{part_num}部分", part_content))
                part_num += 1
            
            start = end
        
        return parts
