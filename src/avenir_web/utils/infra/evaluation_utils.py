# Copyright 2026 Princeton AI for Accelerating Invention Lab
# Author: Aiden Yiliu Li
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# See LICENSE.txt for the full license text.

def looks_like_api_endpoint(url: str) -> bool:
    try:
        if not url:
            return False
        u = url.lower()
        bad = ["/graphql", "/api/", "/api", "/v1/", "/v2/", "/v3/", "/rest/", ".json", ".csv", ".xml", "?format=json", "output=json"]
        for p in bad:
            if p in u:
                return True
        return False
    except Exception:
        return False
