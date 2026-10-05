"""
游戏内体积渲染与体积雾系统 - 验收测试
运行方式：python -m pytest tests/test_volume_rendering.py -q
共 14 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rendering.volume import VolumeRendering


class TestVolumeRendering:
    def setup_method(self):
        self.system = VolumeRendering()

    # ---- 体积渲染：光线步进 + 散射 ----

    def test_case_01(self):
        """render_volume 使用光线步进：多步采样且累计光学深度"""
        result = self.system.render_volume((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
        assert result["ray_marching"] is True
        assert result["steps"] > 1
        assert result["optical_depth"] > 0.0

    def test_case_02(self):
        """render_volume 计算体积散射：有内散射光且透射率下降"""
        result = self.system.render_volume((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
        assert all(c > 0.0 for c in result["color"])
        assert 0.0 < result["transmittance"] < 1.0

    def test_case_03(self):
        """透射率遵循 Beer-Lambert：均匀雾中 T(3L) == T(L)^3"""
        system = VolumeRendering({"noise_strength": 0.0, "fog_falloff": 0.0})
        near = system.render_volume((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), max_dist=50.0)
        far = system.render_volume((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), max_dist=150.0)
        assert far["transmittance"] < near["transmittance"]
        assert abs(far["transmittance"] - near["transmittance"] ** 3) < 1e-6

    # ---- 体积雾：密度 + 噪声 ----

    def test_case_04(self):
        """体积雾有密度：雾内密度为正"""
        density = self.system.volume_fog((0.0, 1.0, 0.0))
        assert density > 0.0

    def test_case_05(self):
        """雾密度随高度衰减：低处比高处浓"""
        low = self.system.volume_fog((3.0, 0.0, 3.0))
        high = self.system.volume_fog((3.0, 20.0, 3.0))
        assert low > high > 0.0

    def test_case_06(self):
        """雾有噪声：同一高度不同位置密度不同（非均匀灰板）"""
        d1 = self.system.volume_fog((0.0, 1.0, 0.0))
        d2 = self.system.volume_fog((7.3, 1.0, 11.9))
        assert d1 != d2

    # ---- 体积光：散射 + 阴影 ----

    def test_case_07(self):
        """体积光有散射：前向散射强于后向（HG 相函数）"""
        forward = self.system.phase_function(1.0)
        backward = self.system.phase_function(-1.0)
        assert forward > backward
        lit = self.system.volumetric_light((0.0, 1.0, 0.0), view_dir=(0.0, 1.0, 0.0))
        unlit = self.system.volumetric_light((0.0, 1.0, 0.0), view_dir=(0.0, -1.0, 0.0))
        assert lit["intensity"][0] > unlit["intensity"][0]

    def test_case_08(self):
        """体积光有阴影：云下点的光强低于云上点"""
        system = VolumeRendering({
            "noise_strength": 0.0, "fog_falloff": 0.0,
            "cloud_base": 20.0, "cloud_top": 40.0,
            "cloud_coverage": 1.0, "cloud_density": 1.0,
        })
        below = system.volumetric_light((0.0, 10.0, 0.0))
        above = system.volumetric_light((0.0, 50.0, 0.0))
        assert below["shadow"] < above["shadow"]
        assert below["intensity"][0] < above["intensity"][0]

    # ---- 体积云：密度场 + 光照 ----

    def test_case_09(self):
        """体积云有密度场：层内有云、层外无云"""
        inside = max(
            self.system.cloud_density_at((x * 3.7, 65.0, z * 3.7))
            for x in range(5) for z in range(5)
        )
        assert inside > 0.0
        assert self.system.cloud_density_at((0.0, 10.0, 0.0)) == 0.0
        assert self.system.cloud_density_at((0.0, 100.0, 0.0)) == 0.0

    def test_case_10(self):
        """体积云有光照：云顶比云底亮"""
        system = VolumeRendering({
            "cloud_coverage": 1.0, "cloud_density": 0.5,
            "cloud_base": 50.0, "cloud_top": 80.0,
        })
        top = system.cloud_lighting((0.5, 78.0, 0.5))
        bottom = system.cloud_lighting((0.5, 52.0, 0.5))
        assert top > bottom

    # ---- 体积阴影：光线步进 + 软阴影 ----

    def test_case_11(self):
        """体积阴影使用光线步进：累计光学深度得到衰减"""
        system = VolumeRendering({
            "cloud_base": 20.0, "cloud_top": 40.0,
            "cloud_coverage": 1.0, "cloud_density": 1.0,
        })
        result = system.volume_shadow((0.0, 10.0, 0.0))
        assert result["steps"] > 1
        assert result["optical_depth"] > 0.0
        assert 0.0 < result["shadow"] < 1.0

    def test_case_12(self):
        """体积阴影是软阴影：穿过云层时连续渐变，存在半影区"""
        system = VolumeRendering({
            "noise_strength": 0.0,
            "cloud_base": 20.0, "cloud_top": 40.0,
            "cloud_coverage": 1.0, "cloud_density": 0.5,
        })
        shadows = [
            system.volume_shadow((0.0, y, 0.0))["shadow"]
            for y in (10.0, 22.0, 26.0, 30.0, 34.0, 38.0, 45.0)
        ]
        assert any(0.05 < s < 0.95 for s in shadows)  # 半影，非硬边
        assert all(b >= a - 1e-9 for a, b in zip(shadows, shadows[1:]))  # 连续单调

    # ---- 综合 ----

    def test_case_13(self):
        """render_volume 综合雾与云：穿过云层的视线透射率更低"""
        system = VolumeRendering({
            "cloud_coverage": 1.0, "cloud_density": 0.8,
            "cloud_base": 50.0, "cloud_top": 80.0,
        })
        through_cloud = system.render_volume(
            (0.0, 10.0, 0.0), (0.0, 1.0, 0.0), max_dist=100.0)
        below_cloud = system.render_volume(
            (0.0, 10.0, 0.0), (1.0, 0.0, 0.0), max_dist=100.0)
        assert through_cloud["transmittance"] < below_cloud["transmittance"]

    def test_case_14(self):
        """update 驱动噪声流动，reset 恢复初始状态"""
        point = (1.0, 2.0, 3.0)
        before = self.system.volume_fog(point)
        self.system.update(1.0)
        after = self.system.volume_fog(point)
        assert before != after
        assert len(self.system._history) == 1
        self.system.reset()
        assert self.system._history == []
        assert self.system.volume_fog(point) == before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
