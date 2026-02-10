"""
市场情绪分析系统
分析市场情绪,包括新闻情绪、社交媒体情绪、期货情绪等
"""

import logging
import re
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class SentimentType(Enum):
    """情绪类型"""
    NEWS = "新闻情绪"
    SOCIAL = "社交媒体情绪"
    FUTURES = "期货市场情绪"
    OPTIONS = "期权情绪"
    TECHNICAL = "技术情绪"
    FUNDAMENTAL = "基本面情绪"

class SentimentLevel(Enum):
    """情绪等级"""
    EXTREME_BULLISH = "极度看涨"
    BULLISH = "看涨"
    NEUTRAL = "中性"
    BEARISH = "看跌"
    EXTREME_BEARISH = "极度看跌"
    MIXED = "混合情绪"

class SourceType(Enum):
    """新闻来源类型"""
    OFFICIAL_MEDIA = "官方媒体"
    FINANCIAL_MEDIA = "财经媒体"
    SOCIAL_MEDIA = "社交媒体"
    FORUM = "论坛"
    BLOG = "博客"
    REGULATORY = "监管公告"

@dataclass
class NewsItem:
    """新闻数据"""
    timestamp: datetime
    title: str
    content: str
    source: str
    source_type: SourceType
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    topics: List[str]
    symbols: List[str]

@dataclass
class SocialMediaItem:
    """社交媒体数据"""
    timestamp: datetime
    content: str
    platform: str  # Weibo, Twitter, Reddit, etc.
    author: str
    followers_count: int
    sentiment_score: float
    confidence: float
    topics: List[str]
    symbols: List[str]
    engagement: Dict[str, int]  # likes, shares, comments

@dataclass
class MarketSentiment:
    """市场情绪"""
    timestamp: datetime
    symbol: str
    overall_sentiment: float  # -1 to 1
    sentiment_level: SentimentLevel
    confidence: float
    news_sentiment: float
    social_sentiment: float
    futures_sentiment: float
    sentiment_trend: str  # improving, stable, deteriorating
    volatility_index: float
    fear_greed_index: float
    market_mood: str  # euphoric, optimistic, neutral, pessimistic, fearful
    key_topics: Dict[str, float]
    sentiment_sources: Dict[str, int]

class SentimentAnalyzer:
    """情绪分析器"""

    def __init__(self, window_size: int = 100):
        """
        初始化情绪分析器

        Args:
            window_size: 分析窗口大小
        """
        self.window_size = window_size
        self.news_history = deque(maxlen=window_size)
        self.social_history = deque(maxlen=window_size)
        self.market_history = deque(maxlen=window_size)

        # 情绪词典
        self.bullish_keywords = {
            '看涨', '上涨', '牛市', '利好', '买入', '增持', '积极', '乐观',
            '突破', '创新高', '强劲', '增长', '繁荣', '看好', '涨势',
            'bullish', 'up', 'buy', 'growth', 'positive', 'optimistic',
            'rally', 'surge', 'boom', 'bull market'
        }

        self.bearish_keywords = {
            '看跌', '下跌', '熊市', '利空', '卖出', '减持', '消极', '悲观',
            '破位', '创新低', '疲软', '衰退', '萧条', '看空', '跌势',
            'bearish', 'down', 'sell', 'decline', 'negative', 'pessimistic',
            'crash', 'slump', 'recession', 'bear market'
        }

        # 话题权重
        self.topic_weights = {
            '宏观经济': 0.3,
            '货币政策': 0.25,
            '政策法规': 0.2,
            '行业动态': 0.15,
            '公司新闻': 0.1
        }

        # 统计数据
        self.total_analyzed = 0
        self.sentiment_statistics = {
            'avg_sentiment': 0,
            'avg_confidence': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0
        }

    def analyze_news_sentiment(self, news_item: NewsItem) -> float:
        """
        分析新闻情绪

        Args:
            news_item: 新闻数据

        Returns:
            情绪分数 (-1 to 1)
        """
        try:
            # 保存到历史
            self.news_history.append(news_item)

            # 文本预处理
            text = (news_item.title + ' ' + news_item.content).lower()

            # 基础情绪分析
            bullish_score = self._count_keywords(text, self.bullish_keywords)
            bearish_score = self._count_keywords(text, self.bearish_keywords)

            # 基于来源的权重调整
            source_weights = {
                SourceType.OFFICIAL_MEDIA: 1.2,
                SourceType.FINANCIAL_MEDIA: 1.0,
                SourceType.SOCIAL_MEDIA: 0.8,
                SourceType.FORUM: 0.7,
                SourceType.BLOG: 0.6,
                SourceType.REGULATORY: 1.5
            }

            weight = source_weights.get(news_item.source_type, 1.0)

            # 计算情绪分数
            if bullish_score + bearish_score > 0:
                sentiment = (bullish_score - bearish_score) / (bullish_score + bearish_score)
            else:
                sentiment = 0

            # 应用权重和置信度
            final_sentiment = sentiment * weight * news_item.confidence

            return max(-1, min(1, final_sentiment))

        except Exception as e:
            logger.error(f"新闻情绪分析失败: {e}")
            return 0

    def analyze_social_sentiment(self, social_item: SocialMediaItem) -> float:
        """
        分析社交媒体情绪

        Args:
            social_item: 社交媒体数据

        Returns:
            情绪分数 (-1 to 1)
        """
        try:
            # 保存到历史
            self.social_history.append(social_item)

            # 文本预处理
            text = social_item.content.lower()

            # 基础情绪分析
            bullish_score = self._count_keywords(text, self.bullish_keywords)
            bearish_score = self._count_keywords(text, self.bearish_keywords)

            # 基于粉丝数量的权重
            follower_weight = min(social_item.followers_count / 10000, 2.0)  # 最大权重2.0

            # 基于互动量的权重
            engagement_weight = 1.0
            total_engagement = sum(social_item.engagement.values())
            if total_engagement > 100:
                engagement_weight = 1.2
            elif total_engagement > 1000:
                engagement_weight = 1.5

            # 计算情绪分数
            if bullish_score + bearish_score > 0:
                sentiment = (bullish_score - bearish_score) / (bullish_score + bearish_score)
            else:
                sentiment = 0

            # 应用权重和置信度
            final_sentiment = sentiment * follower_weight * engagement_weight * social_item.confidence

            return max(-1, min(1, final_sentiment))

        except Exception as e:
            logger.error(f"社交媒体情绪分析失败: {e}")
            return 0

    def analyze_market_sentiment(self, timestamp: datetime, symbol: str,
                                news_sentiment: float, social_sentiment: float) -> MarketSentiment:
        """
        综合分析市场情绪

        Args:
            timestamp: 时间戳
            symbol: 交易品种
            news_sentiment: 新闻情绪
            social_sentiment: 社交媒体情绪

        Returns:
            市场情绪
        """
        try:
            # 计算整体情绪
            overall_sentiment = (news_sentiment * 0.6 + social_sentiment * 0.4)

            # 确定情绪等级
            sentiment_level = self._determine_sentiment_level(overall_sentiment)

            # 计算置信度
            confidence = self._calculate_confidence(news_sentiment, social_sentiment)

            # 分析情绪趋势
            sentiment_trend = self._analyze_sentiment_trend(overall_sentiment)

            # 计算波动率指数
            volatility_index = self._calculate_volatility_index()

            # 计算贪婪恐惧指数
            fear_greed_index = self._calculate_fear_greed_index(overall_sentiment)

            # 确定市场氛围
            market_mood = self._determine_market_mood(overall_sentiment, fear_greed_index)

            # 提取关键话题
            key_topics = self._extract_key_topics()

            # 统计情绪来源
            sentiment_sources = self._count_sentiment_sources()

            return MarketSentiment(
                timestamp=timestamp,
                symbol=symbol,
                overall_sentiment=overall_sentiment,
                sentiment_level=sentiment_level,
                confidence=confidence,
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment,
                futures_sentiment=overall_sentiment,  # 简化处理
                sentiment_trend=sentiment_trend,
                volatility_index=volatility_index,
                fear_greed_index=fear_greed_index,
                market_mood=market_mood,
                key_topics=key_topics,
                sentiment_sources=sentiment_sources
            )

        except Exception as e:
            logger.error(f"市场情绪分析失败: {e}")
            return MarketSentiment(
                timestamp=timestamp,
                symbol=symbol,
                overall_sentiment=0,
                sentiment_level=SentimentLevel.NEUTRAL,
                confidence=0,
                news_sentiment=0,
                social_sentiment=0,
                futures_sentiment=0,
                sentiment_trend='stable',
                volatility_index=0,
                fear_greed_index=50,
                market_mood='neutral',
                key_topics={},
                sentiment_sources={}
            )

    def _count_keywords(self, text: str, keywords: set) -> int:
        """统计关键词数量"""
        count = 0
        for keyword in keywords:
            count += len(re.findall(rf'\b{keyword}\b', text, re.IGNORECASE))
        return count

    def _determine_sentiment_level(self, sentiment: float) -> SentimentLevel:
        """确定情绪等级"""
        if sentiment > 0.6:
            return SentimentLevel.EXTREME_BULLISH
        elif sentiment > 0.2:
            return SentimentLevel.BULLISH
        elif sentiment > -0.2:
            return SentimentLevel.NEUTRAL
        elif sentiment > -0.6:
            return SentimentLevel.BEARISH
        else:
            return SentimentLevel.EXTREME_BEARISH

    def _calculate_confidence(self, news_sentiment: float, social_sentiment: float) -> float:
        """计算置信度"""
        # 如果多个来源一致,置信度更高
        if abs(news_sentiment - social_sentiment) < 0.3:
            confidence = 0.8
        elif abs(news_sentiment - social_sentiment) < 0.6:
            confidence = 0.6
        else:
            confidence = 0.4

        # 基于数据量的调整
        total_items = len(self.news_history) + len(self.social_history)
        if total_items > 50:
            confidence = min(confidence * 1.2, 1.0)
        elif total_items < 10:
            confidence = confidence * 0.8

        return confidence

    def _analyze_sentiment_trend(self, current_sentiment: float) -> str:
        """分析情绪趋势"""
        if len(self.market_history) < 2:
            return 'stable'

        previous_sentiment = self.market_history[-1].overall_sentiment
        sentiment_change = current_sentiment - previous_sentiment

        if sentiment_change > 0.2:
            return 'improving'
        elif sentiment_change < -0.2:
            return 'deteriorating'
        else:
            return 'stable'

    def _calculate_volatility_index(self) -> float:
        """计算波动率指数"""
        if len(self.market_history) < 5:
            return 0

        recent_sentiments = [s.overall_sentiment for s in list(self.market_history)[-5:]]
        avg_sentiment = sum(recent_sentiments) / len(recent_sentiments)

        # 计算标准差
        variance = sum((s - avg_sentiment) ** 2 for s in recent_sentiments) / len(recent_sentiments)
        std_dev = variance ** 0.5

        # 标准化到0-100
        volatility = std_dev * 100
        return min(volatility, 100)

    def _calculate_fear_greed_index(self, sentiment: float) -> float:
        """计算贪婪恐惧指数"""
        # 将情绪分数转换为贪婪恐惧指数(0-100)
        # 1 = 极度贪婪, 0 = 极度恐惧
        fear_greed = (sentiment + 1) * 50  # -1 to 1 -> 0 to 100
        return max(0, min(100, fear_greed))

    def _determine_market_mood(self, sentiment: float, fear_greed_index: float) -> str:
        """确定市场氛围"""
        if sentiment > 0.5 and fear_greed_index > 80:
            return 'euphoric'
        elif sentiment > 0.2 and fear_greed_index > 60:
            return 'optimistic'
        elif -0.2 <= sentiment <= 0.2:
            return 'neutral'
        elif sentiment < -0.2 and fear_greed_index < 40:
            return 'pessimistic'
        else:
            return 'fearful'

    def _extract_key_topics(self) -> Dict[str, float]:
        """提取关键话题"""
        topics = defaultdict(int)

        # 从新闻中提取话题
        for news in self.news_history:
            for topic in news.topics:
                topics[topic] += 1

        # 从社交媒体中提取话题
        for social in self.social_history:
            for topic in social.topics:
                topics[topic] += 1

        # 应用权重并标准化
        total_count = sum(topics.values())
        if total_count == 0:
            return {}

        weighted_topics = {}
        for topic, count in topics.items():
            weight = self.topic_weights.get(topic, 1.0)
            weighted_topics[topic] = (count * weight / total_count) * 100

        # 返回前10个最重要的话题
        sorted_topics = sorted(weighted_topics.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_topics[:10])

    def _count_sentiment_sources(self) -> Dict[str, int]:
        """统计情绪来源"""
        sources = defaultdict(int)

        for news in self.news_history:
            sources[news.source] += 1

        for social in self.social_history:
            sources[social.platform] += 1

        return dict(sources)

    def generate_trading_signals(self, market_sentiment: MarketSentiment) -> List[Dict[str, Any]]:
        """生成交易信号"""
        signals = []

        # 基于情绪的信号
        if market_sentiment.overall_sentiment > 0.6:
            signals.append({
                'signal_type': 'EXTREME_BULLISH',
                'strength': 'STRONG',
                'message': '市场极度看涨,谨慎追高',
                'recommendation': '考虑止盈或逢高做空',
                'sentiment_level': market_sentiment.sentiment_level.value,
                'confidence': market_sentiment.confidence
            })
        elif market_sentiment.overall_sentiment < -0.6:
            signals.append({
                'signal_type': 'EXTREME_BEARISH',
                'strength': 'STRONG',
                'message': '市场极度看跌,可能存在超卖',
                'recommendation': '关注反弹机会',
                'sentiment_level': market_sentiment.sentiment_level.value,
                'confidence': market_sentiment.confidence
            })

        # 基于趋势的信号
        if market_sentiment.sentiment_trend == 'improving':
            signals.append({
                'signal_type': 'SENTIMENT_IMPROVING',
                'strength': 'MEDIUM',
                'message': '市场情绪正在改善',
                'recommendation': '逐步建仓',
                'sentiment_level': market_sentiment.sentiment_level.value,
                'confidence': market_sentiment.confidence
            })
        elif market_sentiment.sentiment_trend == 'deteriorating':
            signals.append({
                'signal_type': 'SENTIMENT_DETERIORATING',
                'strength': 'MEDIUM',
                'message': '市场情绪正在恶化',
                'recommendation': '减仓观望',
                'sentiment_level': market_sentiment.sentiment_level.value,
                'confidence': market_sentiment.confidence
            })

        # 基于波动率的信号
        if market_sentiment.volatility_index > 70:
            signals.append({
                'signal_type': 'HIGH_VOLATILITY',
                'strength': 'MEDIUM',
                'message': '市场波动率过高',
                'recommendation': '谨慎交易,控制仓位',
                'sentiment_level': market_sentiment.sentiment_level.value,
                'confidence': market_sentiment.confidence
            })

        # 基于贪婪恐惧指数的信号
        if market_sentiment.fear_greed_index > 80:
            signals.append({
                'signal_type': 'EXCESSIVE_GREED',
                'strength': 'HIGH',
                'message': '市场贪婪指数过高',
                'recommendation': '考虑减仓',
                'sentiment_level': market_sentiment.sentiment_level.value,
                'confidence': market_sentiment.confidence
            })
        elif market_sentiment.fear_greed_index < 20:
            signals.append({
                'signal_type': 'EXCESSIVE_FEAR',
                'strength': 'HIGH',
                'message': '市场恐惧指数过高',
                'recommendation': '关注抄底机会',
                'sentiment_level': market_sentiment.sentiment_level.value,
                'confidence': market_sentiment.confidence
            })

        return signals

    def get_sentiment_summary(self) -> Dict[str, Any]:
        """获取情绪摘要"""
        if not self.market_history:
            return {}

        latest_sentiment = self.market_history[-1]

        # 统计情绪分布
        sentiment_distribution = {
            SentimentLevel.EXTREME_BULLISH: 0,
            SentimentLevel.BULLISH: 0,
            SentimentLevel.NEUTRAL: 0,
            SentimentLevel.BEARISH: 0,
            SentimentLevel.EXTREME_BEARISH: 0,
            SentimentLevel.MIXED: 0
        }

        for sentiment in self.market_history:
            sentiment_distribution[sentiment.sentiment_level] += 1

        return {
            'timestamp': latest_sentiment.timestamp.isoformat(),
            'symbol': latest_sentiment.symbol,
            'overall_sentiment': latest_sentiment.overall_sentiment,
            'sentiment_level': latest_sentiment.sentiment_level.value,
            'confidence': latest_sentiment.confidence,
            'fear_greed_index': latest_sentiment.fear_greed_index,
            'market_mood': latest_sentiment.market_mood,
            'volatility_index': latest_sentiment.volatility_index,
            'sentiment_trend': latest_sentiment.sentiment_trend,
            'total_analyzed': self.total_analyzed,
            'sentiment_distribution': sentiment_distribution,
            'key_topics': latest_sentiment.key_topics,
            'sentiment_sources': latest_sentiment.sentiment_sources
        }

    def simulate_news_data(self) -> NewsItem:
        """模拟新闻数据"""
        news_templates = [
            {
                'title': '宏观经济数据显示积极信号',
                'content': '最新发布的经济数据显示,GDP增长超出预期,制造业PMI回升,表明经济正在复苏。',
                'sentiment': 0.8
            },
            {
                'title': '政策利好出台',
                'content': '央行宣布降准降息,释放流动性,有利于股市上涨。',
                'sentiment': 0.7
            },
            {
                'title': '行业数据表现疲软',
                'content': '最新行业数据显示,需求下滑,企业盈利预期下调。',
                'sentiment': -0.6
            },
            {
                'title': '监管政策收紧',
                'content': '监管部门加强监管,对相关行业产生负面影响。',
                'sentiment': -0.8
            },
            {
                'title': '中性报告发布',
                'content': '市场分析师认为,当前市场处于平衡状态,上下空间有限。',
                'sentiment': 0.0
            }
        ]

        import random
        template = random.choice(news_templates)

        sources = [
            ('新华社', SourceType.OFFICIAL_MEDIA),
            ('财经网', SourceType.FINANCIAL_MEDIA),
            ('财新网', SourceType.FINANCIAL_MEDIA),
            ('微博', SourceType.SOCIAL_MEDIA),
            ('知乎', SourceType.FORUM)
        ]

        source, source_type = random.choice(sources)

        return NewsItem(
            timestamp=datetime.now(),
            title=template['title'],
            content=template['content'],
            source=source,
            source_type=source_type,
            sentiment_score=template['sentiment'],
            confidence=random.uniform(0.6, 0.95),
            topics=random.sample(list(self.topic_weights.keys()), 2),
            symbols=['SHFE.cu2401', 'DCE.i2403', 'CZCE.MA401']
        )

    def simulate_social_data(self) -> SocialMediaItem:
        """模拟社交媒体数据"""
        social_templates = [
            {
                'content': '今天市场表现不错,值得关注！',
                'sentiment': 0.6
            },
            {
                'content': '担心市场回调,应该减仓了',
                'sentiment': -0.5
            },
            {
                'content': '看到利好消息,准备买入',
                'sentiment': 0.7
            },
            {
                'content': '市场恐慌情绪蔓延,大家要谨慎',
                'sentiment': -0.7
            },
            {
                'content': '观望为主,等待更好的机会',
                'sentiment': 0.0
            }
        ]

        import random
        template = random.choice(social_templates)

        platforms = ['微博', 'Twitter', 'Reddit', '雪球']
        platform = random.choice(platforms)

        return SocialMediaItem(
            timestamp=datetime.now(),
            content=template['content'],
            platform=platform,
            author=f'user_{random.randint(1000, 9999)}',
            followers_count=random.randint(100, 100000),
            sentiment_score=template['sentiment'],
            confidence=random.uniform(0.5, 0.9),
            topics=random.sample(list(self.topic_weights.keys()), 2),
            symbols=['SHFE.cu2401', 'DCE.i2403', 'CZCE.MA401'],
            engagement={
                'likes': random.randint(0, 1000),
                'shares': random.randint(0, 500),
                'comments': random.randint(0, 200)
            }
        )