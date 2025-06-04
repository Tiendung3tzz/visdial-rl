(function(){
  function pad(num, size) {
      var s = "000000000000" + num;
      return s.substr(s.length-size);
  }

  if (!String.prototype.format) {
      String.prototype.format = function() {
          var args = arguments;
          return this.replace(/{(\d+)}/g, function(match, number) {
              return typeof args[number] != 'undefined'
                  ? args[number]
                  : match;  
          });
      };
  }

  $.get('results/finalep6.json', function(data) {
      var image_root = "http://images.cocodataset.org/val2014/";
      
      if (data.opts.sampleWords == 0) {
          $('#heading').html('Encoder: ' + data.opts.encoder
                              + ', Decoder: ' + data.opts.decoder
                              + ', Beam size: ' + data.opts.beamSize
                              + '<br>' + 'Q-Bot: checkpoints/abot_sl_60.vd' //+ data.opts.qbot
                              + '<br>' + 'A-Bot: checkpoints/qbot_sl_60.vd'); //+ data.opts.abot);
      } else {
          $('#heading').html('Encoder: ' + data.opts.encoder
              + ', Decoder: ' + data.opts.decoder + ', Temperature: ' + data.opts.temperature);
      }

      var html = '';
      for (var i in data.data) {
          if (i % 4 == 0) {
              html += "<div class='row'>";
          }

          var image_id = pad(parseInt(data.data[i].image_id), 12);
          var image_src = '{0}COCO_val2014_{1}.jpg'.format(image_root, image_id);

          html += "<div class='col-xs-3'>";
          html += "<img class='col-xs-12' src='" + image_src + "'>";
          html += "<p class='col-xs-12' style='font-weight:400'><span> Image ID: " + image_id + "</span></p>"; // Displaying image_id
          html += "<p class='col-xs-12' style='font-weight:400'><span> Caption: " + data.data[i].caption + "</span></p>";
          html += "<div class='col-xs-12'><ol style='margin-top:10px;'>";

          for (var j = 0; j < 5; j++) {
              html += "<li style='font-weight:400;'><span>" + data.data[i].dialog[j].question + "</span><span>" + data.data[i].dialog[j].answer + "</span></li>";
          }

          html += "</ol></div></div>";

          if (i % 4 == 3) {
              html += "</div><hr>";
          }
      }

      $('#main').html(html);
  });
})();
