const assert = require('assert');
const { formatFunctionResponse } = require('../app/frontend/script.js');

const listResp = {
    name: 'listFiles',
    response: {
        items: [
            { name: 'README.md', type: 'file', path: 'README.md' }
        ]
    }
};
const html = formatFunctionResponse(listResp);
assert(html.includes('<table') && html.includes('README.md'));
console.log('listFiles formatting ok');

const defaultResp = {
    name: 'fetchFiles',
    response: { files: { 'a.txt': 'hello' }, status: 'success' }
};
const html2 = formatFunctionResponse(defaultResp);
assert(html2.includes('{\n') || html2.includes('<pre'));
console.log('default formatting ok');
