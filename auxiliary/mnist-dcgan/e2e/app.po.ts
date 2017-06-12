import { browser, by, element } from 'protractor';

export class MnistDcganPage {
  navigateTo() {
    return browser.get('/');
  }

  getParagraphText() {
    return element(by.css('app-root h1')).getText();
  }
}
