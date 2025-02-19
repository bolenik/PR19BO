#! python
#-*- coding: utf-8 -*-
import unittest
import GKConverter as GKConverter

class GKConverterTest(unittest.TestCase):
	# def testSimpleConversion(self):
    #     (x, y) = GKConverter.convert_GK_to_lat_long(181304,579613)
    #     print('test')

		# self.assertAlmostEqual(50.11526435691097, x)
		# self.assertAlmostEqual(8.687625204011725, y)

	def testSimpleGKtoLatLongConversion(self):
		(x, y) = GKConverter.gauss_krueger_transformation(3477733, 5553274)

		self.assertAlmostEqual(50.1164273930041, x)
		self.assertAlmostEqual(8.6886330000005, y)

	def testHelmertTransformation(self):
		(x, y) = GKConverter.seven_parameter_helmert_transf(50.1164273930041, 8.6886330000005)

		self.assertAlmostEqual(50.11526435691097, x)
		self.assertAlmostEqual(8.687625204011725, y)

	def testWrongInput(self):
		for (right, height) in ((100000, 1000000), (1000000, 100000), (1000000, 1000000)):
			self.assertRaises(ValueError, GKConverter.convert_GK_to_lat_long, right, height)
			self.assertRaises(ValueError, GKConverter.convert_GK_to_lat_long, right, height)

if __name__ == '__main__':
	(x, y) = GKConverter.convert_GK_to_lat_long(579613,181304)
