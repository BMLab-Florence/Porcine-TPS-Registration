"""Unit tests create tiny numerical arrays in memory; no shipped image dataset."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from porcine_tps.registration import (
    TPS, Registration, downsample, grid_cells, pixel_stretch_maps,
    prepare_images, preview_bgr, load_image, save_image,
)
from porcine_tps.landmarks import validate_pairs, write_csv, read_csv, read_legacy_points


class TPSTests(unittest.TestCase):
    def setUp(self):
        self.src = np.array([[8,8],[48,8],[48,39],[8,39],[28,23],[15,29]],dtype=float)
        self.shift = np.array([6.,-4.])

    def test_translation_and_inverse(self):
        target = self.src+self.shift
        fw = TPS.fit(self.src,target)
        inv = TPS.fit(target,self.src)
        points = np.array([[1,4],[20,14],[45,50],[60,25]],dtype=float)
        np.testing.assert_allclose(fw.transform(points),points+self.shift,atol=1e-7)
        np.testing.assert_allclose(inv.transform(fw.transform(points)),points,atol=1e-7)

    def test_affine_non_uniform_scaling(self):
        target = self.src * np.array([1.2,.8]) + np.array([4,3])
        t = TPS.fit(self.src,target)
        sample = np.array([[22.,14.],[37,25.]],dtype=float)
        np.testing.assert_allclose(t.transform(sample),sample * [1.2,.8] + [4,3],atol=1e-7)
        cells = grid_cells(t,(64,48),12)
        self.assertGreater(len(cells),0)
        np.testing.assert_allclose([c.stretch_x for c in cells],1.2,atol=1e-6)
        np.testing.assert_allclose([c.stretch_y for c in cells],.8,atol=1e-6)

    def test_nonrigid_landmark_interpolation(self):
        dest=self.src.copy();dest[4] += np.array([2.,-3.]);dest[5]+=np.array([-1.,2.])
        fitted=TPS.fit(self.src,dest)
        np.testing.assert_allclose(fitted.transform(self.src),dest,atol=1e-6)
        self.assertTrue(np.isfinite(fitted.transform([[200.,200.]])).all())

    def test_reject_degenerate_points(self):
        with self.assertRaisesRegex(ValueError,"four"):
            validate_pairs(self.src[:3],self.src[:3])
        bad = self.src.copy();bad[1]=bad[0]
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            validate_pairs(bad,self.src)
        line = np.array([[0,0],[1,1],[2,2],[3,3]],dtype=float)
        with self.assertRaisesRegex(ValueError,"collinear"):
            validate_pairs(line,line)
        with self.assertRaisesRegex(ValueError,"outside"):
            validate_pairs(self.src,self.src,(15,15),(70,60))

    def test_image_warp_translates_distinguishable_pixels(self):
        """A CLEARED dot at x=20 maps to x=25 on FIXED; inverse mapping samples x-5."""
        cleared=np.zeros((45,65),np.uint8)
        cleared[21,20]=255
        fixed=np.zeros((45,65),np.uint8)
        cp=self.src.copy()
        fp=cp+np.array([5.,0.])
        r=Registration(fixed,cleared,fp,cp)
        warp=r.warp(cleared)
        self.assertEqual(int(warp[21,25]),255)
        self.assertEqual(int(warp[21,20]),0)
        self.assertEqual(warp.dtype,np.uint8)

    def test_uint16_and_output_files(self):
        cleared=np.zeros((45,65),np.uint16);cleared[21,20]=50000
        fixed=np.zeros_like(cleared)
        r=Registration(fixed,cleared,self.src+np.array([5.,0.]),self.src)
        with tempfile.TemporaryDirectory() as folder:
            report = r.export(folder,grid_step=10,pixel_maps=True)
            self.assertLess(report["forward_landmark_error"]["rms_px"],1e-6)
            for name in ("restored.tif","landmarks_used.csv", "stretch_x_cells.csv",
                         "stretch_y_cells.csv","stretch_cells_detailed.csv", "deformed_grid_overlay.png",
                         "stretch_direction_classes.png", "fixed_restored_overlay.png",
                         "fixed_vs_restored.png","stretch_x_pixels.tif","stretch_y_pixels.tif",
                         "registration_report.json"):
                self.assertTrue((Path(folder)/name).exists(),name)
            restored=load_image(Path(folder)/"restored.tif")
            self.assertEqual(restored.dtype,np.uint16)
            self.assertEqual(int(restored[21,25]),50000)
            px = load_image(Path(folder)/"stretch_x_pixels.tif")
            self.assertEqual(px.dtype,np.float32)

    def test_preprocessing_and_image_roundtrip(self):
        sample = np.arange(120*80,dtype=np.uint16).reshape(80,120)
        ff,cc=prepare_images(sample,sample,2,False)
        self.assertEqual(ff.shape,(40,60));self.assertEqual(cc.shape,(40,60))
        self.assertEqual(downsample(sample,1).shape,sample.shape)
        self.assertEqual(preview_bgr(sample).shape,(80,120,3))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"unicode_à_image.tif"
            save_image(path,sample)
            np.testing.assert_array_equal(load_image(path),sample)

    def test_batch_process(self):
        f=np.zeros((45,65),np.uint8)
        c=np.zeros_like(f);c[21,20]=100
        r=Registration(f,c,self.src+np.array([5.,0.]),self.src)
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/"source"; source.mkdir()
            out=Path(folder)/"output"
            save_image(source/"slice_1.tif",c)
            result=r.warp_folder(source,out)
            self.assertEqual(result["registered"],["slice_1.tif"])
            self.assertEqual(result["skipped"],[])
            self.assertEqual(int(load_image(out/"slice_1_restored.tif")[21,25]),100)
            with self.assertRaisesRegex(ValueError,"different"):
                r.warp_folder(source,source)

    def test_pixel_maps_bounds(self):
        fitted=TPS.fit(self.src,self.src)
        cells=grid_cells(fitted,(65,45),11)
        a,b=pixel_stretch_maps(cells,65,45)
        self.assertEqual(a.shape,(45,65))
        self.assertEqual(b.shape,(45,65))
        self.assertTrue(np.any(a>0))


class LandmarkTests(unittest.TestCase):
    def setUp(self):
        self.fixed = np.array([[1,2],[10,2],[10,10],[1,10]],dtype=float)
        self.cleared = self.fixed + [1,0]

    def test_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            filename=Path(folder)/"pts.csv"
            write_csv(filename,self.fixed,self.cleared)
            a,b=read_csv(filename)
            np.testing.assert_array_equal(a,self.fixed)
            np.testing.assert_array_equal(b,self.cleared)

    def test_historical_points_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"historic.points"
            parts=[]
            for group in (self.cleared,self.fixed,self.cleared,self.fixed):
                parts += [str(len(group))]
                for xy in group:parts += [str(xy[0]),str(xy[1])]
            path.write_text("\n".join(parts),encoding="utf-8")
            fixed, cleared=read_legacy_points(path)
            np.testing.assert_array_equal(fixed,self.fixed)
            np.testing.assert_array_equal(cleared,self.cleared)


if __name__ == "__main__":
    unittest.main()
