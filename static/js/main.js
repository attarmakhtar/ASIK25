let chart;

function updateChart(data, ikan, tahunPrediksi=null) {
    let labels = data.map(d => d.tahun);
    let values = data.map(d => d.total_kg);

    // Jika ada input tahun prediksi
    if (tahunPrediksi) {
        labels.push(tahunPrediksi);
        // ambil total_kg_pred dari JSON, kalau ada
        let pred = data.find(d => d.tahun === tahunPrediksi);
        values.push(pred ? pred.total_kg_pred : null);
    }

    if(chart) chart.destroy();

    const ctx = document.getElementById('stokChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: `Stok ${ikan} (kg)`,
                data: values,
                borderColor: '#4CAF50',
                backgroundColor: 'rgba(76, 175, 80, 0.2)',
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

async function loadData() {
    const res = await fetch('/static/grafik_stok_ikan.json');
    const json = await res.json();
    const ikanSelect = document.getElementById('ikanSelect');
    
    // isi dropdown ikan
    let ikans = [...new Set(json.map(d => d.nama_ikan))];
    ikans.forEach(i => {
        let opt = document.createElement('option');
        opt.value = i;
        opt.innerText = i;
        ikanSelect.appendChild(opt);
    });

    // update chart awal
    let dataIkan = json.filter(d => d.nama_ikan === ikans[0]);
    updateChart(dataIkan, ikans[0]);

    // event update
    document.getElementById('updateBtn').addEventListener('click', () => {
        let selectedIkan = ikanSelect.value;
        let tahunPrediksi = document.getElementById('tahunInput').value;
        let dataSelected = json.filter(d => d.nama_ikan === selectedIkan);
        updateChart(dataSelected, selectedIkan, tahunPrediksi ? parseInt(tahunPrediksi) : null);
    });
}

window.onload = loadData;
