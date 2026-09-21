"""One Langfuse configuration; incomplete credentials must fail explicitly."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dreamer'))
import lf

class TestLangfuseConfig(unittest.TestCase):
    def test_explicit_environment_selects_one_complete_project(self):
        config={'LANGFUSE_BASE_URL':'https://dev.example/','LANGFUSE_PUBLIC_KEY':'dev-public','LANGFUSE_SECRET_KEY':'dev-secret'}
        with patch.dict(os.environ,config,clear=True), patch.object(lf,'dotenv_values',return_value={}):
            reader=lf.LangfuseRO()
            self.assertEqual(reader.base,'https://dev.example')
            self.assertEqual(reader.auth,('dev-public','dev-secret'))

    def test_partial_project_fails_instead_of_using_production(self):
        with patch.dict(os.environ,{'LANGFUSE_BASE_URL':'https://dev.example'},clear=True), patch.object(lf,'dotenv_values',return_value={}):
            with self.assertRaisesRegex(ValueError,'together'):
                lf.LangfuseRO()

    def test_missing_configuration_fails(self):
        with patch.dict(os.environ,{},clear=True), patch.object(lf,'dotenv_values',return_value={}):
            with self.assertRaisesRegex(ValueError,'together'):
                lf.LangfuseRO()
