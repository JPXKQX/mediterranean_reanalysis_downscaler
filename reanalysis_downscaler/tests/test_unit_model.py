# -*- coding: utf-8 -*-
import unittest
import reanalysis_downscaler.models.api as api

class TestModelMethods(unittest.TestCase):

    def setUp(self):
        self.meta = deep_api.get_metadata()

    def test_model_metadata_type(self):
        """
        Test that get_metadata() returns dict
        """
        self.assertTrue(type(self.meta) is dict)

    def test_model_metadata_values(self):
        """
        Test that get_metadata() returns right values (subset)
        """
        self.assertEqual(self.meta['name'].lower().replace('-','_'),
                        'reanalysis_downscaler'.lower().replace('-','_'))
        self.assertEqual(self.meta['author'].lower(),
                         'Mario Santa Cruz,Antonio Perez,Javier Diez'.lower())
        self.assertEqual(self.meta['license'].lower(),
                         'Apache 2.0'.lower())


if __name__ == '__main__':
    unittest.main()
