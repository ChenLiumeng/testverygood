import pytest
import pandas as pd
import cntext as ct
from read_couplet_csv import sentiment_analysis, plot_couplet_length
from unittest.mock import patch


@pytest.fixture(scope='class')
def sample_couplet_data():
    """提供测试用的对联数据，包含正常文本、空字符串和超长文本等边缘情况"""
    return pd.DataFrame({
        '上联': [
            '春回大地',          # 正常文本
            '明月松间照',        # 较长文本
            '',                  # 空字符串
            '非常长的上联文本用来测试字数统计功能是否正常工作',  # 超长文本
            None,                # None值
            '   ',               # 空白字符
            '测试特殊字符！@#￥%'
        ],
        '下联': [
            '福满人间',          # 正常文本
            '清泉石上流',        # 较长文本
            '下联无上联',        # 对应空上联
            '对应的下联文本同样需要足够长以测试边界情况',  # 超长文本
            '',                  # 对应None上联
            '',                  # 对应空白上联
            '特殊字符测试&*()_+'
        ]
    })


@pytest.fixture(scope='class')
def sentiment_dict():
    """加载真实的情感分析字典用于测试"""
    return ct.load_pkl_dict('DUTIR.pkl')['DUTIR']


@pytest.mark.usefixtures('sample_couplet_data', 'sentiment_dict')
class TestSentimentAnalysis:
    """情感分析功能测试类"""

    @patch('read_couplet_csv.ct.sentiment')
    def test_sentiment_analysis_normal_case(self, mock_sentiment, sample_couplet_data, sentiment_dict):
        """测试正常情况下的情感分析功能
        验证：能够正确识别积极情感词汇并返回合理结果
        """
        # 模拟情感分析返回结果
        mock_sentiment.return_value = {
            '乐_num': 2, '好_num': 1, '怒_num': 0, '哀_num': 0,
            '惧_num': 0, '恶_num': 0, '惊_num': 0, 'stopword_num': 0,
            'word_num': 5, 'sentence_num': 1
        }
        
        # 选取正常文本的子集
        test_group = sample_couplet_data.iloc[[0, 1]]
        result = sentiment_analysis(test_group)
        
        # 验证返回结果结构和基本数值
        assert isinstance(result, pd.Series)
        assert '乐_num' in result.index
        assert result['乐_num'] == 2  # 匹配模拟返回值
        assert result['怒_num'] == 0
        # 验证情感分析函数被正确调用
        mock_sentiment.assert_called_once()

    def test_sentiment_analysis_empty_text(self, sample_couplet_data):
        """测试空文本的情感分析
        验证：空文本时所有情感数值应为0
        """
        test_group = sample_couplet_data.iloc[[2]]  # 空上联
        result = sentiment_analysis(test_group)
        
        # 验证所有情感数值为0
        for emotion in ['乐_num', '好_num', '怒_num', '哀_num', '惧_num', '恶_num', '惊_num']:
            assert result[emotion] == 0

    @pytest.mark.parametrize('row_index, expected_total', [
        (0, 8),   # 正常文本
        (1, 10),  # 5+5较长文本
        (2, 5),   # 0+5空上联
        (3, 45),  # 24+21超长文本
        (4, 0),   # None+空字符串
        (5, 3),   # 空白字符+空字符串
    ])
    def test_character_count_calculation(self, sample_couplet_data, row_index, expected_total):
        """测试不同情况下的字数统计
        验证：能够正确计算各种边缘情况下的总字数
        """
        row = sample_couplet_data.iloc[row_index]
        shanglian = str(row['上联']) if row['上联'] is not None else ''
        xialian = str(row['下联']) if row['下联'] is not None else ''
        total = len(shanglian) + len(xialian)
        assert total == expected_total


class TestGroupingFunctionality:
    """对联分组功能测试类"""

    def test_group_by_character_count(self, sample_couplet_data):
        """测试按总字数分组功能
        验证：能够正确按上下联总字数进行分组
        """
        # 添加总字数列
        sample_couplet_data['总字数'] = sample_couplet_data.apply(
            lambda row: len(str(row['上联'])) + len(str(row['下联'])), axis=1
        )
        
        # 执行分组
        grouped = sample_couplet_data.groupby('总字数')
        
        # 验证分组结果
        assert 8 in grouped.groups  # 正常文本组
        assert 45 in grouped.groups  # 超长文本组
        assert len(grouped.groups) >= 4  # 至少应有4个不同的组

